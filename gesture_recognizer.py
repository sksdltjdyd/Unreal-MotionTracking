# gesture_recognizer.py - 닌자 게임 인식 시스템 (완전판)

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pythonosc import udp_client, osc_bundle_builder, osc_message_builder
from enum import Enum
import logging
import gesture_stabilizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GestureType(Enum):
    """제스처 타입 정의"""
    NONE = "none"
    FLICK = "flick"
    FIST = "fist"
    PALM_PUSH = "palm_push"
    CROSS_BLOCK = "cross_block" # 원본 Enum에 있었으나 감지 로직에는 명시적으로 사용되지 않음
    CIRCLE = "circle"

class Config:
    """설정값 관리"""
    # OSC 설정
    OSC_IP = "127.0.0.1"
    OSC_PORT = 7000
    
    # MediaPipe 설정
    MAX_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 1
    
    # 카메라 설정
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1
    
    # 제스처 임계값
    FLICK_SPEED_THRESHOLD = 500  # pixels/sec
    FIST_ANGLE_THRESHOLD = 90    # degrees
    PALM_EXTEND_THRESHOLD = 150  # degrees
    CIRCLE_STD_THRESHOLD = 30    # pixels (원의 반지름 균일도)
    CIRCLE_MIN_POINTS = 15       # 원 제스처 판정을 위한 최소 포인트 수
    CIRCLE_MIN_RADIUS_MEAN = 10  # 원으로 간주되기 위한 최소 평균 반지름 (픽셀 단위)
    CIRCLE_MIN_TOTAL_ANGLE = np.pi * 1.5 # 원으로 간주되기 위한 최소 총 각도 변화 (약 270도)

    # 스무딩 설정 (명시적으로 사용되진 않으나 향후 확장 가능)
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30

    # 기본 안정화 장치 설정 (필요시 오버라이드 가능)
    DEFAULT_STABILITY_WINDOW = 0.3    # 초 (제스처 유지 시간)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8  # 최소 신뢰도
    DEFAULT_COOLDOWN_TIME = 0.5       # 초 (동일 제스처 재사용 대기 시간)


class GestureValidator:
    """제스처 유효성 검증 (쿨다운 등)"""
    def __init__(self, min_confidence=0.7, cooldown_time=0.3):
        self.min_confidence = min_confidence
        self.cooldown_time = cooldown_time
        self.last_gesture_time = {} # 손별, 제스처별 마지막 인식 시간
        self.gesture_history = deque(maxlen=10) # 최근 제스처 기록 (디버깅용)
    
    def validate(self, gesture_type, confidence, hand_label):
        """제스처 유효성 검증"""
        current_time = time.time()
        
        # 신뢰도 체크
        if confidence < self.min_confidence:
            return False
        
        # 쿨다운 체크
        gesture_key = f"{gesture_type}_{hand_label}"
        if gesture_key in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_key] < self.cooldown_time:
                return False
        
        # 유효한 제스처
        self.last_gesture_time[gesture_key] = current_time
        self.gesture_history.append({
            "type": gesture_type,
            "hand": hand_label,
            "time": current_time,
            "confidence": confidence
        })
        return True

class NinjaGestureRecognizer:
    """닌자 게임 제스처 인식기"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        # OSC 설정
        self.client = udp_client.SimpleUDPClient(
            osc_ip or Config.OSC_IP, 
            osc_port or Config.OSC_PORT
        )

        # 안정화 모듈 설정
        actual_stabilizer_settings = stabilizer_settings or {}
        try:
            # gesture_stabilizer.py 파일 내에 GestureStabilizer 클래스가 있다고 가정
            self.stabilizer = gesture_stabilizer.GestureStabilizer(
                stability_window=actual_stabilizer_settings.get('stability_window', Config.DEFAULT_STABILITY_WINDOW),
                confidence_threshold=actual_stabilizer_settings.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD),
                cooldown_time=actual_stabilizer_settings.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
            )
            logger.info("GestureStabilizer initialized successfully.")
        except AttributeError as e:
            logger.error(f"Failed to initialize GestureStabilizer from gesture_stabilizer module: {e}. Ensure 'gesture_stabilizer.py' contains 'GestureStabilizer' class.")
            # 안정화 장치 초기화 실패 시 더미(dummy) 안정화 장치 사용 또는 에러 발생 처리
            class DummyStabilizer: # API 호환성을 위한 최소한의 더미 클래스
                def __init__(self, *args, **kwargs): self.stability_window = kwargs.get('stability_window', 0.3)
                def should_send_gesture(self, gesture_type, confidence, hand_label): return True, {"confidence": confidence} # 항상 통과
                def get_statistics(self): return {"current_gesture": "none", "stability_progress": 0}
                def reset_if_idle(self): pass
            self.stabilizer = DummyStabilizer()
            logger.warning("Using a DUMMY stabilizer due to an initialization error with GestureStabilizer.")
        except Exception as e_stab:
            logger.error(f"An unexpected error occurred while initializing GestureStabilizer: {e_stab}")
            #  위와 동일한 더미 사용
            class DummyStabilizer:
                def __init__(self, *args, **kwargs): self.stability_window = kwargs.get('stability_window', 0.3)
                def should_send_gesture(self, gesture_type, confidence, hand_label): return True, {"confidence": confidence}
                def get_statistics(self): return {"current_gesture": "none", "stability_progress": 0}
                def reset_if_idle(self): pass
            self.stabilizer = DummyStabilizer()
            logger.warning("Using a DUMMY stabilizer due to an unexpected initialization error.")

        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
            model_complexity=Config.MODEL_COMPLEXITY
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 랜드마크 인덱스 정의
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20  # PINKY_TIP은 여기에 정의되어 있습니다.
                             # "pinky 손가락 인덱스도 정의되어 있지 않아"라는 문제가 이 변수를 지칭한다면,
                             # 해당 변수는 정상적으로 정의되어 있습니다.
                             # 만약 다른 pinky 관련 랜드마크(예: MCP, PIP, DIP)가 필요하다면 별도 추가가 필요합니다.
        
        # 제스처 유효성 검증기 (주로 쿨다운 관리)
        self.validator = GestureValidator()
        
        # 상태 추적용 변수
        self.prev_landmarks = {"Left": None, "Right": None} # 이전 프레임의 손 랜드마크 (플릭 감지용)
        self.prev_time = time.time() # 이전 프레임 처리 시간 (dt 계산용)
        self.position_history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)} # 손 위치 기록 (스무딩 등에 활용 가능)
        self.circle_points = {"Left": deque(maxlen=20), "Right": deque(maxlen=20)} # 원 그리기 판정용 포인트 기록

        logger.info(f"Ninja Gesture Recognizer initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

    def calculate_distance(self, p1, p2):
        """두 점 사이의 유클리드 거리 계산"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_angle(self, p1, p2, p3):
        """세 점(p1-p2-p3 순서)으로 이루어진 각도 계산 (p2가 꼭지점)"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0: # 벡터 길이가 0이면 각도 계산 불가
            return 0.0 

        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # clip으로 값 범위 보정
        return np.degrees(angle)

    def calculate_finger_angles(self, landmarks):
        """손가락 굴곡 각도 계산 (주로 PIP 관절 기준)"""
        angles = {}
        # 각 손가락의 관절 인덱스 (MCP, PIP, DIP, TIP 순서이나, 각도 계산에는 주로 MCP-PIP-DIP 또는 PIP-DIP-TIP 사용)
        finger_joints_indices = {
            'thumb': [1, 2, 3, 4],   # 엄지는 구조가 다름 (CMC, MCP, IP, TIP)
            'index': [5, 6, 7, 8],   # 검지 (MCP, PIP, DIP, TIP)
            'middle': [9, 10, 11, 12], # 중지
            'ring': [13, 14, 15, 16],  # 약지
            'pinky': [17, 18, 19, 20]  # 새끼손가락
        }
        
        for finger, joints in finger_joints_indices.items():
            # 보통 손가락 굽힘은 PIP 관절(두 번째 관절)을 기준으로 측정
            # landmarks[joints[1]] (PIP)가 꼭지점이 되도록 p1, p2, p3 설정
            p1 = np.array([landmarks[joints[0]].x, landmarks[joints[0]].y]) # MCP
            p2 = np.array([landmarks[joints[1]].x, landmarks[joints[1]].y]) # PIP
            p3 = np.array([landmarks[joints[2]].x, landmarks[joints[2]].y]) # DIP
            
            angles[finger] = self.calculate_angle(p1, p2, p3)
        return angles

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지"""
        angles = self.calculate_finger_angles(landmarks)
        closed_fingers = 0
        # 엄지를 제외한 네 손가락이 충분히 굽혀졌는지 확인
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles and angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                closed_fingers += 1
        
        # 보통 4개 손가락 모두 닫혀야 주먹으로 간주하나, 여기서는 3개 이상으로 설정됨
        is_fist = closed_fingers >= 3 
        confidence = closed_fingers / 4.0 # 0.0 ~ 1.0 사이의 신뢰도
        return is_fist, confidence

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기(플릭) 감지 (주로 검지)"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0 # 이전 랜드마크 없음

        current_time = time.time()
        dt = current_time - self.prev_time # 이전 프레임과의 시간 간격
        if dt == 0:
            return False, None, 0.0

        # 검지 끝(INDEX_TIP)의 현재 위치와 이전 위치
        curr_index_tip_lm = current_landmarks[self.INDEX_TIP]
        prev_index_tip_lm = self.prev_landmarks[hand_label][self.INDEX_TIP]
        
        # 이미지 좌표계로 변환 (x, y)
        curr_pos = np.array([curr_index_tip_lm.x * img_width, curr_index_tip_lm.y * img_height])
        prev_pos = np.array([prev_index_tip_lm.x * img_width, prev_index_tip_lm.y * img_height])
        
        distance = self.calculate_distance(curr_pos, prev_pos)
        velocity = distance / dt # 속도 (pixels/sec)

        if velocity > Config.FLICK_SPEED_THRESHOLD:
            direction_vector = curr_pos - prev_pos
            norm_direction = np.linalg.norm(direction_vector)
            if norm_direction == 0: # 위치 변화가 없으면 방향 없음
                return False, None, 0.0
            
            direction_normalized = direction_vector / norm_direction
            
            # 플릭 시 검지가 펴져 있는지 확인
            finger_angles = self.calculate_finger_angles(current_landmarks)
            if 'index' in finger_angles and finger_angles['index'] > 120: # 120도 이상 펴져있음
                return True, direction_normalized.tolist(), velocity
        
        return False, None, 0.0

    def detect_palm_push(self, landmarks, hand_label): # hand_label은 현재 미사용
        """손바닥 밀기 감지"""
        finger_angles = self.calculate_finger_angles(landmarks)
        extended_fingers = 0
        # 엄지를 제외한 네 손가락이 펴져 있는지 확인
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in finger_angles and finger_angles[finger] > Config.PALM_EXTEND_THRESHOLD:
                extended_fingers += 1
        
        # 3개 이상 손가락이 펴져 있고 (보통 4개), 손바닥이 카메라 쪽으로 향하는 움직임 감지
        if extended_fingers >= 3:
            # 손목(WRIST)과 손바닥 중심(MIDDLE_FINGER_MCP, landmark 9)의 z값 차이로 밀기 감지 시도
            # 이 방식은 손의 기울기에 매우 민감할 수 있음
            palm_center_lm = landmarks[self.MIDDLE_TIP - 3] # MIDDLE_FINGER_MCP (landmark 9)
            wrist_lm = landmarks[self.WRIST]
            
            # z값은 카메라로부터의 거리를 나타내며, 값이 작을수록 카메라에 가까움
            # 미는 동작은 손바닥이 손목보다 카메라에 더 가까워지는(z값이 작아지는) 경향
            z_diff = wrist_lm.z - palm_center_lm.z # 양수면 손바닥이 더 가까움
            
            # 특정 임계값 이상으로 z 차이가 나면 푸시로 간주 (0.05는 실험적으로 조정 필요)
            if z_diff > 0.05: 
                confidence = min(z_diff * 10, 1.0) # z_diff가 클수록 신뢰도 높게 (최대 1.0)
                return True, confidence
        
        return False, 0.0

    def detect_circle(self, landmarks, hand_label, img_width, img_height):
        """원 그리기 감지 (검지 끝 사용)"""
        index_tip_lm = landmarks[self.INDEX_TIP]
        current_pos = np.array([index_tip_lm.x * img_width, index_tip_lm.y * img_height])
        
        self.circle_points[hand_label].append(current_pos)
        
        if len(self.circle_points[hand_label]) >= Config.CIRCLE_MIN_POINTS:
            points = np.array(self.circle_points[hand_label])
            center = np.mean(points, axis=0) # 포인트들의 평균 중심
            distances_from_center = np.linalg.norm(points - center, axis=1) # 각 포인트와 중심 간 거리
            
            mean_radius = np.mean(distances_from_center)
            std_dev_radius = np.std(distances_from_center)

            # 반지름 표준편차가 작고 (균일한 원), 평균 반지름이 너무 작지 않으면 원으로 판단
            if std_dev_radius < Config.CIRCLE_STD_THRESHOLD and mean_radius > Config.CIRCLE_MIN_RADIUS_MEAN:
                total_angle_change = 0
                for i in range(len(points) - 1):
                    p_curr = points[i]
                    p_next = points[i+1]
                    
                    angle_segment = np.arctan2(p_next[1] - center[1], p_next[0] - center[0]) - \
                                    np.arctan2(p_curr[1] - center[1], p_curr[0] - center[0])
                    
                    # 각도 변화량을 -pi ~ pi 범위로 정규화
                    if angle_segment > np.pi: angle_segment -= 2 * np.pi
                    if angle_segment < -np.pi: angle_segment += 2 * np.pi
                    total_angle_change += angle_segment

                # 총 각도 변화량이 특정 임계값(예: 270도)을 넘으면 원으로 최종 판단
                if abs(total_angle_change) > Config.CIRCLE_MIN_TOTAL_ANGLE:
                    # 이미지 좌표계 (y축 아래로 증가) 기준: total_angle_change > 0 이면 반시계(ccw)
                    direction = "ccw" if total_angle_change > 0 else "cw"
                    self.circle_points[hand_label].clear() # 포인트 기록 초기화
                    return True, direction
        
        return False, None

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        """단일 손에 대한 통합 제스처 인식"""
        landmarks = hand_landmarks_obj.landmark # MediaPipe Landmark list
        height, width = img_shape[:2]
        
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0} # 기본 신뢰도

        # 제스처 감지 순서 (우선순위 고려 필요)
        # 1. 플릭 (빠른 움직임이므로 우선 감지)
        is_flick, flick_dir, flick_speed = self.detect_flick(landmarks, hand_label, width, height)
        if is_flick:
            current_gesture = GestureType.FLICK
            # 플릭은 명확한 움직임이므로 신뢰도를 높게 설정 가능
            gesture_data = {"direction": flick_dir, "speed": flick_speed, "confidence": 0.9}
        else:
            # 2. 주먹 (플릭이 아닐 경우)
            is_fist, fist_conf = self.detect_fist(landmarks)
            if is_fist:
                current_gesture = GestureType.FIST
                gesture_data = {"confidence": fist_conf}
            else:
                # 3. 손바닥 밀기 (플릭, 주먹이 아닐 경우)
                is_push, push_conf = self.detect_palm_push(landmarks, hand_label)
                if is_push:
                    current_gesture = GestureType.PALM_PUSH
                    gesture_data = {"confidence": push_conf}
        
        # 4. 원 그리기 (다른 제스처와 동시에 발생 가능성 고려)
        # 현재 로직은 원이 감지되면 이전 제스처를 덮어씀.
        # 만약 원 제스처가 다른 제스처와 독립적이거나 우선순위가 낮다면,
        # `if current_gesture == GestureType.NONE and is_circle:` 와 같이 조건 추가 가능
        is_circle, circle_dir = self.detect_circle(landmarks, hand_label, width, height)
        if is_circle:
            current_gesture = GestureType.CIRCLE
            # 원 그리기는 비교적 명확하므로 신뢰도를 적절히 설정
            gesture_data = {"direction": circle_dir, "confidence": 0.85} 

        # 다음 프레임의 플릭 감지를 위해 현재 랜드마크 저장
        self.prev_landmarks[hand_label] = landmarks 
        return current_gesture, gesture_data

    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value

            # 제스처 유효성 검증 (쿨다운 등) - 안정화 장치 통과 후 추가 검증
            if not self.validator.validate(gesture_type_str, confidence, hand_label):
                # logger.debug(f"Gesture {gesture_type_str} for {hand_label} failed validation (cooldown).")
                return

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            # 제스처 타입 메시지
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type")
            msg_builder.add_arg(gesture_type_str)
            bundle_builder.add_content(msg_builder.build())
            
            # 신뢰도 메시지
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence")
            msg_builder.add_arg(float(confidence))
            bundle_builder.add_content(msg_builder.build())

            # 방향 정보 (플릭, 원)
            if "direction" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                direction_val = gesture_data["direction"]
                if isinstance(direction_val, list) and len(direction_val) == 2: # 플릭 (x, y 벡터)
                    msg_builder.add_arg(float(direction_val[0]))
                    msg_builder.add_arg(float(direction_val[1]))
                else: # 원 (문자열 "cw" 또는 "ccw")
                    msg_builder.add_arg(str(direction_val))
                bundle_builder.add_content(msg_builder.build())

            # 속도 정보 (플릭)
            if "speed" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                msg_builder.add_arg(float(gesture_data["speed"]))
                bundle_builder.add_content(msg_builder.build())
            
            # 손 구분 메시지
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
            msg_builder.add_arg(hand_label) # "Left" or "Right"
            bundle_builder.add_content(msg_builder.build())
            
            # 번들 전송
            self.client.send(bundle_builder.build())
            logger.info(f"OSC Sent: {gesture_type_str} ({hand_label}), Conf: {confidence:.2f}")

        except Exception as e:
            logger.error(f"OSC 전송 중 오류 발생: {e}")

    def send_hand_state(self, hand_count):
        """손 감지 상태 및 개수 OSC 전송"""
        try:
            self.client.send_message("/ninja/hand/detected", 1 if hand_count > 0 else 0)
            self.client.send_message("/ninja/hand/count", hand_count)
        except Exception as e:
            logger.error(f"손 상태 OSC 전송 중 오류 발생: {e}")

    def process_frame(self, frame_input):
        """단일 프레임 처리 및 제스처 인식 후 결과 반환"""
        frame_to_draw_on = frame_input.copy() # 원본 프레임 수정을 피하기 위해 복사본 사용
        debug_messages_for_frame = [] # 현재 프레임의 디버그 메시지 리스트

        try:
            # BGR -> RGB 변환 및 MediaPipe 처리
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label # "Left" or "Right"
                    
                    # 손 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 제스처 인식 (raw)
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape
                    )
                    
                    # 안정화 필터 적용
                    # GestureStabilizer.should_send_gesture 는 (bool, dict) 를 반환한다고 가정
                    should_send, stabilized_gesture_data = self.stabilizer.should_send_gesture(
                        raw_gesture_type.value, # 제스처 타입 문자열 전달
                        raw_gesture_data.get("confidence", 0.0),
                        hand_label
                    )
                    
                    if should_send and raw_gesture_type != GestureType.NONE:
                        # OSC 전송 (안정화된 제스처 정보 사용)
                        # stabilized_gesture_data가 OSC 전송에 필요한 모든 정보를 포함해야 함
                        # 여기서는 원본 raw_gesture_data를 사용 (필요시 stabilized_gesture_data 활용)
                        self.send_gesture_osc(raw_gesture_type, raw_gesture_data, hand_label)
                        debug_messages_for_frame.append(f"{hand_label}: {raw_gesture_type.value} \u2713 (Sent)")
                    else:
                        # 안정화 대기 중인 제스처 정보 표시
                        stabilizer_stats = self.stabilizer.get_statistics()
                        # gesture_stabilizer.py의 get_statistics() 반환 형식에 맞춰야 함
                        pending_gesture = stabilizer_stats.get("current_gesture", "none")
                        
                        if pending_gesture != "none" and pending_gesture == raw_gesture_type.value:
                            stability_progress = stabilizer_stats.get("stability_progress", 0)
                            # 안정화 창 지속 시간 (GestureStabilizer 인스턴스에서 가져오거나 Config 사용)
                            window_duration = getattr(self.stabilizer, 'stability_window', Config.DEFAULT_STABILITY_WINDOW)
                            progress_percent = min(stability_progress / window_duration if window_duration > 0 else 0, 1.0) * 100
                            debug_messages_for_frame.append(f"{hand_label}: {pending_gesture} ({progress_percent:.0f}%)")
                        elif raw_gesture_type != GestureType.NONE:
                            # 안정화 필터에 걸렸지만 아직 대기 상태는 아닌 경우 (예: 너무 짧게 유지)
                            debug_messages_for_frame.append(f"{hand_label}: {raw_gesture_type.value} (Raw)")
            else:
                # 손이 감지되지 않으면 안정화 장치 리셋 (필요시)
                self.stabilizer.reset_if_idle()

            # 손 감지 상태 OSC 전송
            hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            self.send_hand_state(hand_count)

        except Exception as e:
            logger.error(f"프레임 처리 중 심각한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            debug_messages_for_frame.append(f"Error processing: {e}")

        self.prev_time = time.time() # 현재 프레임 처리 시간 기록
        return frame_to_draw_on, debug_messages_for_frame

    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        logger.info("NinjaGestureRecognizer resources cleaned up.")


class NinjaMasterHandTracker:
    """닌자 마스터 메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings_override=None):
        # 제스처 인식기 초기화 (안정화 장치 설정 전달)
        self.gesture_recognizer = NinjaGestureRecognizer(
            osc_ip=osc_ip,
            osc_port=osc_port,
            stabilizer_settings=stabilizer_settings_override
        )
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0) # 0번 카메라 사용 (필요시 변경)
        if not self.cap.isOpened():
            error_msg = "웹캠을 열 수 없습니다. 카메라 연결 상태 및 권한을 확인하세요."
            logger.error(error_msg)
            raise IOError(error_msg) # 프로그램 중단을 위해 IOError 발생

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE) # 버퍼 크기 줄여 지연 감소 시도
        
        # FPS 계산용 변수
        self.fps_counter = deque(maxlen=Config.FPS_BUFFER_SIZE)
        self.last_time = time.time() # FPS 계산 위한 이전 시간 초기화
        
        # 디버그 모드
        self.debug_mode = True
        logger.info("Ninja Master Hand Tracker initialized.")

    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        time_difference = current_time - self.last_time
        
        if time_difference > 0: # 0으로 나누기 방지
            fps = 1.0 / time_difference
            self.fps_counter.append(fps)
        
        self.last_time = current_time # 현재 시간을 다음 계산을 위해 저장
        
        return np.mean(self.fps_counter) if len(self.fps_counter) > 0 else 0.0

    def draw_debug_info(self, frame, fps, debug_messages_list):
        """디버그 정보(FPS, 제스처 상태 등)를 프레임에 그리기"""
        # FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 제스처 및 기타 디버그 메시지 표시
        y_offset = 70
        for message in debug_messages_list:
            cv2.putText(frame, message, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30
            
        # 안내 문구
        cv2.putText(frame, "Ninja Master - Gesture Recognition", (10, Config.CAMERA_HEIGHT - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug", (10, Config.CAMERA_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def run(self):
        """메인 실행 루프"""
        logger.info("닌자 마스터 제스처 인식 시작... 'q'를 눌러 종료")
        
        try:
            while True:
                # 웹캠에서 프레임 읽기
                success, frame_from_camera = self.cap.read()
                if not success:
                    logger.error("웹캠에서 프레임을 읽을 수 없습니다. 루프를 종료합니다.")
                    break
                
                # 좌우 반전 (거울 모드)
                current_frame_flipped = cv2.flip(frame_from_camera, 1)
                
                # 현재 프레임에 대한 출력 변수 초기화
                # process_frame에서 오류 발생 시 이 기본값을 사용하게 됨
                processed_display_frame = current_frame_flipped 
                current_debug_messages = [] 
                
                # 제스처 인식 처리 (오류는 process_frame 내부에서 로깅)
                processed_display_frame, current_debug_messages = self.gesture_recognizer.process_frame(current_frame_flipped)
                
                # FPS 계산
                current_fps = self.calculate_fps()
                
                # 디버그 정보 표시 (활성화 시)
                if self.debug_mode:
                    self.draw_debug_info(processed_display_frame, current_fps, current_debug_messages)
                
                # 화면에 결과 프레임 표시
                cv2.imshow("Ninja Master", processed_display_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF # 1ms 대기, 키 입력 없으면 -1 반환
                if key == ord('q'): # 'q' 입력 시 종료
                    logger.info("'q' 키 입력됨. 프로그램을 종료합니다.")
                    break
                elif key == ord('d'): # 'd' 입력 시 디버그 모드 토글
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e_main:
            logger.error(f"메인 실행 루프에서 예외 발생: {e_main}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup() # 루프 종료 시 리소스 정리

    def cleanup(self):
        """리소스 정리"""
        logger.info("프로그램 종료 중... 리소스를 정리합니다.")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup()
        logger.info("닌자 마스터 제스처 인식 시스템이 종료되었습니다.")


# 테스트 모드 (test_osc_communication.py 파일 필요)
def test_mode():
    """OSC 통신 테스트 모드"""
    try:
        from test_osc_communication import OSCTester # 이 파일과 클래스가 존재해야 함
        
        logger.info("=== OSC 통신 테스트 모드 시작 ===")
        tester = OSCTester()
        tester.start_server() # OSCTester에 이 메서드가 있다고 가정

        while True:
            print("\n테스트 옵션:")
            print("1. 모든 제스처 테스트 (가상 전송)")
            print("2. 손 추적 테스트 (OSC 메시지 확인)")
            print("3. 게임플레이 시뮬레이션 (연속 제스처)")
            print("4. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                tester.test_all_gestures() # OSCTester에 이 메서드가 있다고 가정
            elif choice == "2":
                tester.test_hand_tracking() # OSCTester에 이 메서드가 있다고 가정
            elif choice == "3":
                tester.simulate_gameplay() # OSCTester에 이 메서드가 있다고 가정
            elif choice == "4":
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        
        if hasattr(tester, 'stop_server'): # 서버 중지 메서드가 있다면 호출
             tester.stop_server()
        logger.info("OSC 테스트 모드 종료.")

    except ImportError:
        logger.error("'test_osc_communication.py' 또는 'OSCTester' 클래스를 찾을 수 없습니다. 테스트 모드를 실행할 수 없습니다.")
    except Exception as e_test_mode:
        logger.error(f"테스트 모드 실행 중 오류 발생: {e_test_mode}")


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 "test"가 주어졌는지 확인
    is_test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    if is_test_mode:
        test_mode()
    else:
        # 일반 실행 시 사용할 커스텀 안정화 장치 설정
        custom_stabilizer_settings = {
            "stability_window": 0.4,    # 더 긴 안정화 시간
            "confidence_threshold": 0.85, # 더 높은 최소 신뢰도
            "cooldown_time": 0.7        # 더 긴 쿨다운
        }
        
        try:
            # NinjaMasterHandTracker에 안정화 설정 전달
            tracker = NinjaMasterHandTracker(
                stabilizer_settings_override=custom_stabilizer_settings
            )
            tracker.run()
        except IOError as e_io_cam: # 웹캠 열기 실패 시
            logger.critical(f"프로그램 시작 실패 (IOError): {e_io_cam}. 카메라를 사용할 수 없습니다.")
        except KeyboardInterrupt: # 사용자가 Ctrl+C로 종료 시
            logger.info("\n프로그램이 사용자에 의해 중단되었습니다.")
        except Exception as e_global_run: # 기타 예기치 않은 오류
            logger.critical(f"프로그램 실행 중 치명적인 오류 발생: {e_global_run}")
            import traceback
            traceback.print_exc()