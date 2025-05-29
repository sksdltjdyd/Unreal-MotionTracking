# gesture_recognizer.py - 닌자 게임 인식 시스템 (수정 완료 버전)

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
    CROSS_BLOCK = "cross_block"
    CIRCLE = "circle"


class Config:
    """설정값 관리"""
    # OSC 설정
    OSC_IP = "127.0.0.1"
    OSC_PORT = 7000
    
    # MediaPipe 설정 - 인식률 향상
    MAX_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5  # 더 쉽게 손 감지
    MIN_TRACKING_CONFIDENCE = 0.3   # 더 부드러운 추적
    MODEL_COMPLEXITY = 1
    
    # 카메라 설정
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1
    
    # 제스처 임계값 - 모두 완화
    FLICK_SPEED_THRESHOLD = 300     # 더 느린 움직임도 인식
    FIST_ANGLE_THRESHOLD = 100      # 덜 굽혀도 인식
    PALM_EXTEND_THRESHOLD = 140     # 덜 펴도 인식
    CIRCLE_STD_THRESHOLD = 50       
    CIRCLE_MIN_POINTS = 10          
    CIRCLE_MIN_RADIUS_MEAN = 15     
    CIRCLE_MIN_TOTAL_ANGLE = np.pi  # 180도만 그려도 인식
    
    # 기본 안정화 설정 (추가됨!)
    DEFAULT_STABILITY_WINDOW = 0.3      # 제스처 유지 시간
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8   # 최소 신뢰도
    DEFAULT_COOLDOWN_TIME = 0.5         # 재사용 대기 시간
    
    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30


class GestureValidator:
    """제스처 유효성 검증 (쿨다운 등)"""
    def __init__(self, min_confidence=0.7, cooldown_time=0.3):
        self.min_confidence = min_confidence
        self.cooldown_time = cooldown_time
        self.last_gesture_time = {}
        self.gesture_history = deque(maxlen=10)
    
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

        # 안정화 모듈 설정 - 수정된 부분
        if stabilizer_settings is None:
            stabilizer_settings = {}
        
        try:
            self.stabilizer = gesture_stabilizer.GestureStabilizer(
                stability_window=stabilizer_settings.get('stability_window', Config.DEFAULT_STABILITY_WINDOW),
                confidence_threshold=stabilizer_settings.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD),
                cooldown_time=stabilizer_settings.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
            )
            logger.info(f"GestureStabilizer initialized: window={self.stabilizer.stability_window}s, "
                       f"threshold={self.stabilizer.confidence_threshold}, "
                       f"cooldown={self.stabilizer.cooldown_time}s")
        except Exception as e:
            logger.error(f"Failed to initialize GestureStabilizer: {e}")
            # 간단한 대체 안정화 클래스
            class SimpleStabilizer:
                def __init__(self, **kwargs):
                    self.stability_window = kwargs.get('stability_window', 0.3)
                    self.confidence_threshold = kwargs.get('confidence_threshold', 0.8)
                    self.cooldown_time = kwargs.get('cooldown_time', 0.5)
                    self.last_gesture_time = {}
                    self.current_gesture = "none"
                    self.current_gesture_start = 0
                
                def should_send_gesture(self, gesture_type, confidence, hand_label):
                    current_time = time.time()
                    
                    # None 제스처는 무시
                    if gesture_type == "none":
                        self.current_gesture = "none"
                        return False, None
                    
                    # 신뢰도 체크
                    if confidence < self.confidence_threshold:
                        return False, None
                    
                    # 새로운 제스처 시작
                    if gesture_type != self.current_gesture:
                        self.current_gesture = gesture_type
                        self.current_gesture_start = current_time
                        return False, None
                    
                    # 안정화 시간 체크
                    if current_time - self.current_gesture_start < self.stability_window:
                        return False, None
                    
                    # 쿨다운 체크
                    gesture_key = f"{gesture_type}_{hand_label}"
                    if gesture_key in self.last_gesture_time:
                        if current_time - self.last_gesture_time[gesture_key] < self.cooldown_time:
                            return False, None
                    
                    # 제스처 전송
                    self.last_gesture_time[gesture_key] = current_time
                    return True, {"confidence": confidence}
                
                def get_statistics(self):
                    return {
                        "current_gesture": self.current_gesture,
                        "stability_progress": time.time() - self.current_gesture_start if self.current_gesture != "none" else 0
                    }
                
                def reset_if_idle(self):
                    self.current_gesture = "none"
                    self.current_gesture_start = 0
            
            self.stabilizer = SimpleStabilizer(**stabilizer_settings)
            logger.warning("Using SimpleStabilizer as fallback")

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
        self.PINKY_TIP = 20
        
        # 제스처 유효성 검증기
        self.validator = GestureValidator(
            min_confidence=0.5,  # 더 낮은 신뢰도 허용
            cooldown_time=0.2    # 더 짧은 쿨다운
        )
        
        # 상태 추적용 변수
        self.prev_landmarks = {"Left": None, "Right": None}
        self.prev_time = time.time()
        self.position_history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}
        self.circle_points = {"Left": deque(maxlen=20), "Right": deque(maxlen=20)}

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
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0 

        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_finger_angles(self, landmarks):
        """손가락 굴곡 각도 계산"""
        angles = {}
        finger_joints_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger, joints in finger_joints_indices.items():
            p1 = np.array([landmarks[joints[0]].x, landmarks[joints[0]].y])
            p2 = np.array([landmarks[joints[1]].x, landmarks[joints[1]].y])
            p3 = np.array([landmarks[joints[2]].x, landmarks[joints[2]].y])
            
            angles[finger] = self.calculate_angle(p1, p2, p3)
        return angles

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지 - 개선 버전"""
        angles = self.calculate_finger_angles(landmarks)
        
        # 각 손가락의 굽힘 정도를 점수화
        finger_scores = []
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                # 각도가 작을수록 (더 굽혀질수록) 높은 점수
                score = max(0, (Config.FIST_ANGLE_THRESHOLD - angles[finger]) / Config.FIST_ANGLE_THRESHOLD)
                finger_scores.append(score)
        
        if len(finger_scores) >= 3:  # 최소 3개 손가락 데이터가 있어야 함
            avg_score = sum(finger_scores) / len(finger_scores)
            
            # 평균 점수가 0.5 이상이면 주먹으로 인식
            is_fist = avg_score > 0.5
            confidence = min(avg_score + 0.3, 1.0)  # 신뢰도 부스트
            
            return is_fist, confidence
        
        return False, 0.0

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지 - 개선 버전"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0 or dt > 0.5:  # 0.5초 이상 차이나면 무시
            return False, None, 0.0
        
        # 검지와 중지 둘 다 체크 (더 유연한 인식)
        for finger_tip in [self.INDEX_TIP, self.MIDDLE_TIP]:
            curr_tip = current_landmarks[finger_tip]
            prev_tip = self.prev_landmarks[hand_label][finger_tip]
            
            curr_pos = np.array([curr_tip.x * img_width, curr_tip.y * img_height])
            prev_pos = np.array([prev_tip.x * img_width, prev_tip.y * img_height])
            
            distance = self.calculate_distance(curr_pos, prev_pos)
            velocity = distance / dt
            
            if velocity > Config.FLICK_SPEED_THRESHOLD:
                direction_vector = curr_pos - prev_pos
                norm_direction = np.linalg.norm(direction_vector)
                
                if norm_direction > 0:
                    direction_normalized = direction_vector / norm_direction
                    
                    # 손가락이 어느 정도만 펴져 있어도 인식
                    finger_angles = self.calculate_finger_angles(current_landmarks)
                    finger_name = 'index' if finger_tip == self.INDEX_TIP else 'middle'
                    
                    if finger_name in finger_angles and finger_angles[finger_name] > 100:  # 120 → 100
                        # 속도에 따른 신뢰도 계산
                        confidence = min(0.7 + (velocity - Config.FLICK_SPEED_THRESHOLD) / 1000, 1.0)
                        return True, direction_normalized.tolist(), velocity
        
        return False, None, 0.0

    def detect_palm_push(self, landmarks, hand_label):
        """손바닥 밀기 감지 - 개선 버전"""
        finger_angles = self.calculate_finger_angles(landmarks)
        
        # 각 손가락의 펴짐 정도 점수화
        extended_score = 0
        finger_count = 0
        
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in finger_angles:
                # 각도가 클수록 (더 펴질수록) 높은 점수
                score = max(0, (finger_angles[finger] - 100) / 80)  # 100도 이상부터 점수
                extended_score += score
                finger_count += 1
        
        if finger_count >= 3:
            avg_extended = extended_score / finger_count
            
            # 평균 점수가 0.5 이상이면 손바닥이 펴진 것으로 판단
            if avg_extended > 0.5:
                # z축 차이로 밀기 동작 감지
                palm_center = landmarks[9]  # 중지 MCP
                wrist = landmarks[self.WRIST]
                
                # z값 차이 (음수일수록 손바닥이 카메라 쪽으로 향함)
                z_diff = palm_center.z - wrist.z
                
                # 손목 각도도 고려 (손목이 펴져있는지)
                wrist_angle = self.calculate_angle(
                    [landmarks[0].x, landmarks[0].y],  # 손목
                    [landmarks[5].x, landmarks[5].y],  # 검지 MCP
                    [landmarks[17].x, landmarks[17].y]  # 새끼 MCP
                )
                
                # z축 차이가 작거나 손목이 펴져있으면 푸시로 인식
                if z_diff < -0.02 or wrist_angle > 150:
                    confidence = min(0.7 + avg_extended * 0.3, 1.0)
                    return True, confidence
        
        return False, 0.0

    def detect_circle(self, landmarks, hand_label, img_width, img_height):
        """원 그리기 감지 - 개선 버전"""
        # 검지 또는 손 전체 중심점 사용
        index_tip = landmarks[self.INDEX_TIP]
        
        # 손바닥 중심점도 계산 (더 안정적인 추적)
        palm_center_x = (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3
        palm_center_y = (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
        
        # 검지가 충분히 펴져있을 때는 검지 사용, 아니면 손바닥 중심 사용
        finger_angles = self.calculate_finger_angles(landmarks)
        if 'index' in finger_angles and finger_angles['index'] > 120:
            current_pos = np.array([index_tip.x * img_width, index_tip.y * img_height])
        else:
            current_pos = np.array([palm_center_x * img_width, palm_center_y * img_height])
        
        # 너무 가까운 포인트는 스킵
        if len(self.circle_points[hand_label]) > 0:
            last_pos = self.circle_points[hand_label][-1]
            if self.calculate_distance(current_pos, last_pos) < 3:
                return False, None
        
        self.circle_points[hand_label].append(current_pos)
        
        # 충분한 포인트가 모였을 때 분석
        if len(self.circle_points[hand_label]) >= Config.CIRCLE_MIN_POINTS:
            points = np.array(self.circle_points[hand_label])
            
            # 간단한 원 판정: 시작점으로 돌아왔는지 확인
            start_point = points[0]
            current_point = points[-1]
            
            # 중심점 계산
            center = np.mean(points, axis=0)
            
            # 평균 반지름
            distances = np.linalg.norm(points - center, axis=1)
            mean_radius = np.mean(distances)
            
            # 시작점과 현재점의 거리
            closing_distance = self.calculate_distance(start_point, current_point)
            
            # 조건: 충분히 돌았고 시작점 근처로 돌아왔으면 원으로 인식
            if mean_radius > Config.CIRCLE_MIN_RADIUS_MEAN:
                # 각도 변화 계산 (간단히)
                total_angle = 0
                for i in range(1, len(points)):
                    v1 = points[i-1] - center
                    v2 = points[i] - center
                    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                    if angle > np.pi: angle -= 2*np.pi
                    if angle < -np.pi: angle += 2*np.pi
                    total_angle += angle
                
                # 180도 이상 돌았으면 원으로 인식
                if abs(total_angle) > Config.CIRCLE_MIN_TOTAL_ANGLE:
                    # 시작점 근처로 돌아왔거나 충분히 많은 포인트가 있으면
                    if closing_distance < mean_radius or len(points) > 15:
                        direction = "ccw" if total_angle > 0 else "cw"
                        self.circle_points[hand_label].clear()
                        return True, direction
            
            # 포인트가 너무 많으면 오래된 것 제거
            if len(self.circle_points[hand_label]) > 20:
                for _ in range(5):
                    self.circle_points[hand_label].popleft()
        
        return False, None

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        """단일 손에 대한 통합 제스처 인식"""
        landmarks = hand_landmarks_obj.landmark
        height, width = img_shape[:2]
        
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # 제스처 감지 순서 (우선순위 고려)
        # 1. 플릭 (빠른 움직임이므로 우선 감지)
        is_flick, flick_dir, flick_speed = self.detect_flick(landmarks, hand_label, width, height)
        if is_flick:
            current_gesture = GestureType.FLICK
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
        is_circle, circle_dir = self.detect_circle(landmarks, hand_label, width, height)
        if is_circle:
            current_gesture = GestureType.CIRCLE
            gesture_data = {"direction": circle_dir, "confidence": 0.85}

        # 다음 프레임의 플릭 감지를 위해 현재 랜드마크 저장
        self.prev_landmarks[hand_label] = landmarks
        return current_gesture, gesture_data

    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value

            # 제스처 유효성 검증 (쿨다운 등)
            if not self.validator.validate(gesture_type_str, confidence, hand_label):
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
                if isinstance(direction_val, list) and len(direction_val) == 2:
                    msg_builder.add_arg(float(direction_val[0]))
                    msg_builder.add_arg(float(direction_val[1]))
                else:
                    msg_builder.add_arg(str(direction_val))
                bundle_builder.add_content(msg_builder.build())

            # 속도 정보 (플릭)
            if "speed" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                msg_builder.add_arg(float(gesture_data["speed"]))
                bundle_builder.add_content(msg_builder.build())
            
            # 손 구분 메시지
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
            msg_builder.add_arg(hand_label)
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
        frame_to_draw_on = frame_input.copy()
        debug_messages_for_frame = []

        try:
            # BGR -> RGB 변환 및 MediaPipe 처리
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label
                    
                    # 손 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 제스처 인식 (raw)
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape
                    )
                    
                    # 안정화 필터 적용
                    should_send, stabilized_gesture_data = self.stabilizer.should_send_gesture(
                        raw_gesture_type.value,
                        raw_gesture_data.get("confidence", 0.0),
                        hand_label
                    )
                    
                    if should_send and raw_gesture_type != GestureType.NONE:
                        # OSC 전송
                        self.send_gesture_osc(raw_gesture_type, raw_gesture_data, hand_label)
                        debug_messages_for_frame.append(f"{hand_label}: {raw_gesture_type.value} ✓ (Sent)")
                    else:
                        # 안정화 대기 중인 제스처 정보 표시
                        stabilizer_stats = self.stabilizer.get_statistics()
                        pending_gesture = stabilizer_stats.get("current_gesture", "none")
                        
                        if pending_gesture != "none" and pending_gesture == raw_gesture_type.value:
                            stability_progress = stabilizer_stats.get("stability_progress", 0)
                            window_duration = getattr(self.stabilizer, 'stability_window', Config.DEFAULT_STABILITY_WINDOW)
                            progress_percent = min(stability_progress / window_duration if window_duration > 0 else 0, 1.0) * 100
                            debug_messages_for_frame.append(f"{hand_label}: {pending_gesture} ({progress_percent:.0f}%)")
                        elif raw_gesture_type != GestureType.NONE:
                            debug_messages_for_frame.append(f"{hand_label}: {raw_gesture_type.value} (Raw)")
            else:
                # 손이 감지되지 않으면 안정화 장치 리셋
                self.stabilizer.reset_if_idle()

            # 손 감지 상태 OSC 전송
            hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            self.send_hand_state(hand_count)

        except Exception as e:
            logger.error(f"프레임 처리 중 심각한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            debug_messages_for_frame.append(f"Error processing: {e}")

        self.prev_time = time.time()
        return frame_to_draw_on, debug_messages_for_frame

    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        logger.info("NinjaGestureRecognizer resources cleaned up.")


class NinjaMasterHandTracker:
    """닌자 마스터 메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings_override=None):
        # 제스처 인식기 초기화
        self.gesture_recognizer = NinjaGestureRecognizer(
            osc_ip=osc_ip,
            osc_port=osc_port,
            stabilizer_settings=stabilizer_settings_override
        )
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_msg = "웹캠을 열 수 없습니다. 카메라 연결 상태 및 권한을 확인하세요."
            logger.error(error_msg)
            raise IOError(error_msg)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE)
        
        # FPS 계산용 변수
        self.fps_counter = deque(maxlen=Config.FPS_BUFFER_SIZE)
        self.last_time = time.time()
        
        # 디버그 모드
        self.debug_mode = True
        logger.info("Ninja Master Hand Tracker initialized.")

    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        time_difference = current_time - self.last_time
        
        if time_difference > 0:
            fps = 1.0 / time_difference
            self.fps_counter.append(fps)
        
        self.last_time = current_time
        
        return np.mean(self.fps_counter) if len(self.fps_counter) > 0 else 0.0

    def draw_debug_info(self, frame, fps, debug_messages_list):
        """디버그 정보를 프레임에 그리기"""
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
                processed_display_frame = current_frame_flipped
                current_debug_messages = []
                
                # 제스처 인식 처리
                processed_display_frame, current_debug_messages = self.gesture_recognizer.process_frame(current_frame_flipped)
                
                # FPS 계산
                current_fps = self.calculate_fps()
                
                # 디버그 정보 표시 (활성화 시)
                if self.debug_mode:
                    self.draw_debug_info(processed_display_frame, current_fps, current_debug_messages)
                
                # 화면에 결과 프레임 표시
                cv2.imshow("Ninja Master", processed_display_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("'q' 키 입력됨. 프로그램을 종료합니다.")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e_main:
            logger.error(f"메인 실행 루프에서 예외 발생: {e_main}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        logger.info("프로그램 종료 중... 리소스를 정리합니다.")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup()
        logger.info("닌자 마스터 제스처 인식 시스템이 종료되었습니다.")


# 테스트 모드
def test_mode():
    """OSC 통신 테스트 모드"""
    try:
        from test_osc_communication import OSCTester
        
        logger.info("=== OSC 통신 테스트 모드 시작 ===")
        tester = OSCTester()
        tester.start_server()

        while True:
            print("\n테스트 옵션:")
            print("1. 모든 제스처 테스트 (가상 전송)")
            print("2. 손 추적 테스트 (OSC 메시지 확인)")
            print("3. 게임플레이 시뮬레이션 (연속 제스처)")
            print("4. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                tester.test_all_gestures()
            elif choice == "2":
                tester.test_hand_tracking()
            elif choice == "3":
                tester.simulate_gameplay()
            elif choice == "4":
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        
        if hasattr(tester, 'stop_server'):
             tester.stop_server()
        logger.info("OSC 테스트 모드 종료.")

    except ImportError:
        logger.error("'test_osc_communication.py' 또는 'OSCTester' 클래스를 찾을 수 없습니다.")
    except Exception as e_test_mode:
        logger.error(f"테스트 모드 실행 중 오류 발생: {e_test_mode}")


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 "test"가 주어졌는지 확인
    is_test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    if is_test_mode:
        test_mode()
    else:
        # 안정화 설정을 더 관대하게 조정
        custom_stabilizer_settings = {
            "stability_window": 0.2,      # 더 빠른 반응
            "confidence_threshold": 0.6,   # 더 낮은 신뢰도 허용
            "cooldown_time": 0.3          # 더 짧은 쿨다운
        }
        
        try:
            tracker = NinjaMasterHandTracker(
                stabilizer_settings_override=custom_stabilizer_settings
            )
            tracker.run()
        except IOError as e:
            logger.critical(f"프로그램 시작 실패: {e}")
        except KeyboardInterrupt:
            logger.info("\n프로그램이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            logger.critical(f"프로그램 실행 중 치명적인 오류 발생: {e}")
            import traceback
            traceback.print_exc()