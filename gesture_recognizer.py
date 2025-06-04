# gesture_recognizer.py - 닌자 게임 인식 시스템 (심플 3제스처 버전 + 위치 트래킹)

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pythonosc import udp_client, osc_bundle_builder, osc_message_builder
from enum import Enum
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureType(Enum):
    """제스처 타입 정의 - 3개로 단순화"""
    NONE = "none"
    FLICK = "flick"          # 표창 던지기
    FIST = "fist"            # 공격 막기
    PALM_PUSH = "palm_push"  # 진동파


class Config:
    """설정값 관리"""
    # OSC 설정
    OSC_IP = "127.0.0.1"
    OSC_PORT = 7000
    
    # MediaPipe 설정 - 인식률 향상
    MAX_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.3
    MIN_TRACKING_CONFIDENCE = 0.2
    MODEL_COMPLEXITY = 1
    
    # 카메라 설정
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1
    
    # 웹캠 위치 보정 (머리 위)
    CAMERA_ANGLE_CORRECTION = 15
    Y_OFFSET_CORRECTION = 0.1
    
    # 제스처 임계값 - 3개 제스처에 최적화
    FLICK_SPEED_THRESHOLD = 120      # 더 쉬운 플릭 인식
    FIST_ANGLE_THRESHOLD = 110       # 더 관대한 주먹 인식
    PALM_EXTEND_THRESHOLD = 130      # 5손가락 펴기 인식 완화
    
    # 안정화 설정 - 빠른 게임플레이용
    DEFAULT_STABILITY_WINDOW = 0.15   # 빠른 반응
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_COOLDOWN_TIME = 0.3      # 짧은 쿨다운
    
    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30
    
    # 노이즈 필터링
    MOVEMENT_THRESHOLD = 3
    GESTURE_CHANGE_THRESHOLD = 0.2
    
    # 위치 트래킹 설정 (새로 추가)
    POSITION_LEFT_THRESHOLD = 0.33    # 화면의 40% 이하는 좌측
    POSITION_RIGHT_THRESHOLD = 0.66   # 화면의 60% 이상은 우측
    POSITION_TRACKING_SMOOTHING = 0.8 # 위치 스무딩 계수


class SimpleStabilizer:
    """단순화된 제스처 안정화"""
    def __init__(self, **kwargs):
        self.stability_window = kwargs.get('stability_window', Config.DEFAULT_STABILITY_WINDOW)
        self.confidence_threshold = kwargs.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        self.cooldown_time = kwargs.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
        self.last_gesture_time = {}
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer = deque(maxlen=5)
        self.last_sent_gesture = "none"
    
    def should_send_gesture(self, gesture_type, confidence, hand_label):
        current_time = time.time()
        
        if gesture_type == "none":
            return False, None
        
        # 신뢰도 체크
        if confidence < self.confidence_threshold:
            return False, None
        
        self.gesture_buffer.append(gesture_type)
        
        # 버퍼의 60% 이상이 같은 제스처면 인식
        if len(self.gesture_buffer) >= 3:
            gesture_count = {}
            for g in self.gesture_buffer:
                gesture_count[g] = gesture_count.get(g, 0) + 1
            
            most_common = max(gesture_count, key=gesture_count.get)
            if gesture_count[most_common] >= len(self.gesture_buffer) * 0.6:
                gesture_type = most_common
            else:
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
        self.gesture_buffer.clear()
        self.last_sent_gesture = gesture_type
        
        return True, {"confidence": confidence}
    
    def get_statistics(self):
        return {
            "current_gesture": self.current_gesture,
            "stability_progress": time.time() - self.current_gesture_start if self.current_gesture != "none" else 0
        }
    
    def reset_if_idle(self):
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer.clear()


class NinjaGestureRecognizer:
    """닌자 게임 제스처 인식기 - 3제스처 + 위치 트래킹 버전"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        # OSC 설정
        self.client = udp_client.SimpleUDPClient(
            osc_ip or Config.OSC_IP, 
            osc_port or Config.OSC_PORT
        )

        # 안정화 모듈 설정
        if stabilizer_settings is None:
            stabilizer_settings = {}
        
        self.stabilizer = SimpleStabilizer(**stabilizer_settings)
        logger.info(f"Simple Stabilizer initialized with settings: {stabilizer_settings}")

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
        
        # 상태 추적용 변수
        self.prev_landmarks = {"Left": None, "Right": None}
        self.prev_time = time.time()
        self.smoothed_landmarks = {"Left": None, "Right": None}
        
        # 위치 추적용 변수 (새로 추가)
        self.hand_positions = {"Left": "center", "Right": "center"}
        self.smoothed_hand_x = {"Left": 0.5, "Right": 0.5}
        
        # 웹캠 각도 보정을 위한 변수
        self.angle_correction_matrix = self._create_angle_correction_matrix()

        logger.info(f"Ninja Gesture Recognizer (3 Gestures + Position Tracking) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

    def _create_angle_correction_matrix(self):
        """웹캠 각도 보정 매트릭스 생성"""
        angle_rad = np.radians(Config.CAMERA_ANGLE_CORRECTION)
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    def _correct_landmark_position(self, landmark):
        """웹캠 위치에 따른 랜드마크 보정"""
        corrected_y = landmark.y - Config.Y_OFFSET_CORRECTION
        return landmark.x, corrected_y, landmark.z

    def _smooth_landmarks(self, current_landmarks, hand_label):
        """랜드마크 스무딩 적용"""
        if self.smoothed_landmarks[hand_label] is None:
            self.smoothed_landmarks[hand_label] = current_landmarks
            return current_landmarks
        
        alpha = 0.7
        smoothed = []
        for i, (curr, prev) in enumerate(zip(current_landmarks, self.smoothed_landmarks[hand_label])):
            smoothed_x = alpha * curr.x + (1 - alpha) * prev.x
            smoothed_y = alpha * curr.y + (1 - alpha) * prev.y
            smoothed_z = alpha * curr.z + (1 - alpha) * prev.z
            
            class SmoothLandmark:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            
            smoothed.append(SmoothLandmark(smoothed_x, smoothed_y, smoothed_z))
        
        self.smoothed_landmarks[hand_label] = smoothed
        return smoothed

    def calculate_distance(self, p1, p2):
        """두 점 사이의 유클리드 거리 계산"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_angle(self, p1, p2, p3):
        """세 점으로 이루어진 각도 계산"""
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

    def calculate_hand_position(self, landmarks, hand_label):
        """손의 화면상 위치(좌/중앙/우) 판단"""
        # 손목과 중지 MCP의 평균 위치를 손의 중심으로 계산
        wrist_x = landmarks[self.WRIST].x
        middle_mcp_x = landmarks[9].x  # 중지 MCP
        
        # 두 점의 평균으로 손의 중심 X 좌표 계산
        hand_center_x = (wrist_x + middle_mcp_x) / 2
        
        # 스무딩 적용
        self.smoothed_hand_x[hand_label] = (
            Config.POSITION_TRACKING_SMOOTHING * self.smoothed_hand_x[hand_label] + 
            (1 - Config.POSITION_TRACKING_SMOOTHING) * hand_center_x
        )
        
        smoothed_x = self.smoothed_hand_x[hand_label]
        
        # 위치 판단
        if smoothed_x < Config.POSITION_LEFT_THRESHOLD:
            position = "left"
        elif smoothed_x > Config.POSITION_RIGHT_THRESHOLD:
            position = "right"
        else:
            position = "center"
        
        self.hand_positions[hand_label] = position
        
        return position, smoothed_x

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지 - 공격 막기"""
        angles = self.calculate_finger_angles(landmarks)
        
        bent_fingers = 0
        total_fingers = 0
        
        # 엄지를 제외한 4개 손가락 체크
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                total_fingers += 1
                if angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                    bent_fingers += 1
        
        # 3개 이상 손가락이 구부러지면 주먹
        if total_fingers > 0 and bent_fingers >= 3:
            confidence = 0.7 + (bent_fingers / total_fingers) * 0.3
            return True, confidence
        
        return False, 0.0

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지 - 표창 던지기 (위치 정보 포함)"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0, "center"
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0 or dt > 0.5:
            return False, None, 0.0, "center"
        
        # 손 위치 계산
        position, _ = self.calculate_hand_position(current_landmarks, hand_label)
        
        # 검지와 중지 체크
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP]
        best_flick = None
        best_velocity = 0
        
        for finger_tip in finger_tips:
            curr_tip = current_landmarks[finger_tip]
            prev_tip = self.prev_landmarks[hand_label][finger_tip]
            
            curr_x, curr_y, _ = self._correct_landmark_position(curr_tip)
            prev_x, prev_y, _ = self._correct_landmark_position(prev_tip)
            
            curr_pos = np.array([curr_x * img_width, curr_y * img_height])
            prev_pos = np.array([prev_x * img_width, prev_y * img_height])
            
            distance = self.calculate_distance(curr_pos, prev_pos)
            
            if distance < Config.MOVEMENT_THRESHOLD:
                continue
            
            velocity = distance / dt
            
            if velocity > Config.FLICK_SPEED_THRESHOLD and velocity > best_velocity:
                direction_vector = curr_pos - prev_pos
                norm_direction = np.linalg.norm(direction_vector)
                
                if norm_direction > 0:
                    direction_normalized = direction_vector / norm_direction
                    
                    # 방향을 4방향으로 단순화
                    angle = np.arctan2(direction_normalized[1], direction_normalized[0])
                    angle_deg = np.degrees(angle)
                    
                    if -45 <= angle_deg <= 45:
                        simplified_direction = [1.0, 0.0]  # 오른쪽
                    elif angle_deg > 45 and angle_deg <= 135:
                        simplified_direction = [0.0, 1.0]  # 아래
                    elif angle_deg < -45 and angle_deg >= -135:
                        simplified_direction = [0.0, -1.0]  # 위
                    else:
                        simplified_direction = [-1.0, 0.0]  # 왼쪽
                    
                    # 손가락 펴짐 확인
                    finger_angles = self.calculate_finger_angles(current_landmarks)
                    finger_name = 'index' if finger_tip == self.INDEX_TIP else 'middle'
                    
                    if finger_name in finger_angles and finger_angles[finger_name] > 120:
                        best_flick = simplified_direction
                        best_velocity = velocity
        
        if best_flick:
            confidence = min(0.7 + (best_velocity - Config.FLICK_SPEED_THRESHOLD) / 500, 1.0)
            return True, best_flick, best_velocity, position
        
        return False, None, 0.0, position

    def detect_palm_push(self, landmarks, hand_label):
        """손바닥 밀기 감지 - 진동파 (위치 정보 포함)"""
        # 손 위치 계산
        position, _ = self.calculate_hand_position(landmarks, hand_label)
        
        finger_angles = self.calculate_finger_angles(landmarks)
        
        # 모든 손가락이 펴져있는지 확인
        extended_fingers = 0
        total_fingers = 0
        
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in finger_angles:
                total_fingers += 1
                # 엄지는 다른 기준 적용
                threshold = 90 if finger == 'thumb' else Config.PALM_EXTEND_THRESHOLD
                if finger_angles[finger] > threshold:
                    extended_fingers += 1
        
        # 5개 손가락 모두 펴져있으면 palm push
        if total_fingers >= 5 and extended_fingers >= 5:
            # 손바닥이 앞을 향하는지 추가 확인
            wrist_z = landmarks[self.WRIST].z
            middle_mcp_z = landmarks[9].z
            
            # 손바닥이 카메라를 향하고 있으면 신뢰도 증가
            z_diff = middle_mcp_z - wrist_z
            base_confidence = 0.8
            
            if z_diff < 0:  # 손바닥이 카메라를 향함
                confidence = min(base_confidence + abs(z_diff) * 0.5, 1.0)
            else:
                confidence = base_confidence
                
            return True, confidence, position
        
        return False, 0.0, position

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        """통합 제스처 인식 - 위치 정보 포함"""
        landmarks = self._smooth_landmarks(hand_landmarks_obj.landmark, hand_label)
        height, width = img_shape[:2]
        
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # 우선순위에 따른 제스처 인식
        # 1. 플릭 (표창 던지기) - 가장 우선순위 높음
        is_flick, flick_dir, flick_speed, flick_position = self.detect_flick(
            landmarks, hand_label, width, height
        )
        if is_flick:
            current_gesture = GestureType.FLICK
            gesture_data = {
                "direction": flick_dir, 
                "speed": flick_speed, 
                "confidence": 0.85,
                "action": "throw_shuriken",
                "position": flick_position  # 위치 정보 추가
            }
        else:
            # 2. 주먹 (공격 막기) - 위치 정보 불필요
            is_fist, fist_conf = self.detect_fist(landmarks)
            if is_fist:
                current_gesture = GestureType.FIST
                gesture_data = {
                    "confidence": fist_conf,
                    "action": "block_attack"
                }
            else:
                # 3. 손바닥 밀기 (진동파)
                is_push, push_conf, push_position = self.detect_palm_push(landmarks, hand_label)
                if is_push:
                    current_gesture = GestureType.PALM_PUSH
                    gesture_data = {
                        "confidence": push_conf,
                        "action": "shock_wave",
                        "position": push_position  # 위치 정보 추가
                    }

        self.prev_landmarks[hand_label] = hand_landmarks_obj.landmark
        return current_gesture, gesture_data

    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송 (위치 정보 포함)"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value
            action = gesture_data.get("action", "")
            position = gesture_data.get("position", "center")  # 위치 정보 추출

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            # 제스처 타입
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type")
            msg_builder.add_arg(gesture_type_str)
            bundle_builder.add_content(msg_builder.build())
            
            # 게임 액션
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/action")
            msg_builder.add_arg(action)
            bundle_builder.add_content(msg_builder.build())
            
            # 신뢰도
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence")
            msg_builder.add_arg(float(confidence))
            bundle_builder.add_content(msg_builder.build())

            # 위치 정보 (flick과 palm_push에만 해당)
            if gesture_type_str in ["flick", "palm_push"] and "position" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position")
                msg_builder.add_arg(position)
                bundle_builder.add_content(msg_builder.build())
                
                # 위치별 세부 액션 정보
                position_action = f"{action}_{position}"
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position_action")
                msg_builder.add_arg(position_action)
                bundle_builder.add_content(msg_builder.build())

            # 방향 정보 (플릭용)
            if "direction" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                direction_val = gesture_data["direction"]
                if isinstance(direction_val, list) and len(direction_val) == 2:
                    msg_builder.add_arg(float(direction_val[0]))
                    msg_builder.add_arg(float(direction_val[1]))
                bundle_builder.add_content(msg_builder.build())

            # 속도 정보 (플릭용)
            if "speed" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                msg_builder.add_arg(float(gesture_data["speed"]))
                bundle_builder.add_content(msg_builder.build())
            
            # 손 구분
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
            msg_builder.add_arg(hand_label)
            bundle_builder.add_content(msg_builder.build())
            
            self.client.send(bundle_builder.build())
            
            # 로그 메시지 개선
            log_msg = f"OSC Sent: {gesture_type_str} ({action})"
            if gesture_type_str in ["flick", "palm_push"]:
                log_msg += f" @ {position.upper()}"
            log_msg += f" - {hand_label}, Conf: {confidence:.2f}"
            logger.info(log_msg)

        except Exception as e:
            logger.error(f"OSC 전송 중 오류 발생: {e}")

    def send_hand_state(self, hand_count):
        """손 감지 상태 전송"""
        try:
            self.client.send_message("/ninja/hand/detected", 1 if hand_count > 0 else 0)
            self.client.send_message("/ninja/hand/count", hand_count)
        except Exception as e:
            logger.error(f"손 상태 OSC 전송 중 오류 발생: {e}")

    def process_frame(self, frame_input):
        """프레임 처리"""
        frame_to_draw_on = frame_input.copy()
        debug_messages_for_frame = []

        try:
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label
                    
                    # 손 그리기
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 제스처 인식
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape
                    )
                    
                    # 안정화
                    should_send, stabilized_gesture_data = self.stabilizer.should_send_gesture(
                        raw_gesture_type.value,
                        raw_gesture_data.get("confidence", 0.0),
                        hand_label
                    )
                    
                    if should_send and raw_gesture_type != GestureType.NONE:
                        self.send_gesture_osc(raw_gesture_type, raw_gesture_data, hand_label)
                        action = raw_gesture_data.get("action", "")
                        position = raw_gesture_data.get("position", "")
                        debug_msg = f"{hand_label}: {raw_gesture_type.value} ({action})"
                        if position:
                            debug_msg += f" @ {position.upper()}"
                        debug_msg += " ✓"
                        debug_messages_for_frame.append(debug_msg)
                    else:
                        stabilizer_stats = self.stabilizer.get_statistics()
                        pending_gesture = stabilizer_stats.get("current_gesture", "none")
                        
                        if pending_gesture != "none":
                            progress = stabilizer_stats.get("stability_progress", 0)
                            progress_percent = min(progress / self.stabilizer.stability_window * 100, 100)
                            debug_messages_for_frame.append(
                                f"{hand_label}: {pending_gesture} ({progress_percent:.0f}%)"
                            )
            else:
                self.stabilizer.reset_if_idle()

            hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            self.send_hand_state(hand_count)

        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
            debug_messages_for_frame.append(f"Error: {e}")

        self.prev_time = time.time()
        return frame_to_draw_on, debug_messages_for_frame

    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        logger.info("Gesture recognizer cleaned up.")


class NinjaMasterHandTracker:
    """메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings_override=None):
        self.gesture_recognizer = NinjaGestureRecognizer(
            osc_ip=osc_ip,
            osc_port=osc_port,
            stabilizer_settings=stabilizer_settings_override
        )
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_msg = "웹캠을 열 수 없습니다."
            logger.error(error_msg)
            raise IOError(error_msg)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE)
        
        self.fps_counter = deque(maxlen=Config.FPS_BUFFER_SIZE)
        self.last_time = time.time()
        self.debug_mode = True
        
        logger.info("Ninja Master Hand Tracker (3 Gestures + Position) initialized.")

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
        """디버그 정보 그리기 (위치 정보 포함)"""
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 화면 3등분 가이드라인 그리기
        height, width = frame.shape[:2]
        left_line = int(width * Config.POSITION_LEFT_THRESHOLD)
        right_line = int(width * Config.POSITION_RIGHT_THRESHOLD)
        
        # 세로 구분선 (점선 효과)
        for y in range(0, height, 20):
            cv2.line(frame, (left_line, y), (left_line, y + 10), (100, 100, 100), 2)
            cv2.line(frame, (right_line, y), (right_line, y + 10), (100, 100, 100), 2)
        
        # 영역 라벨
        cv2.putText(frame, "LEFT", (left_line // 2 - 20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "CENTER", (width // 2 - 30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (right_line + (width - right_line) // 2 - 25, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        
        # 제스처 정보
        y_offset = 90
        for message in debug_messages_list:
            color = (0, 255, 255) if "✓" in message else (255, 255, 0)
            cv2.putText(frame, message, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y_offset += 30
        
        # 제스처 가이드 - 위치 정보 추가
        guide_y = height - 180
        cv2.putText(frame, "=== 3 Core Gestures + Position ===", (10, guide_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        gestures = [
            ("1. FLICK", "Quick finger movement", "Throw Shuriken (L/C/R)", (255, 100, 100)),
            ("2. FIST", "Close your hand", "Block Attack", (100, 255, 100)),
            ("3. PALM PUSH", "Open all 5 fingers", "Shock Wave (L/C/R)", (100, 100, 255))
        ]
        
        for i, (gesture, desc, action, color) in enumerate(gestures):
            y_pos = guide_y + 30 + (i * 25)
            cv2.putText(frame, f"{gesture}: {desc} = {action}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # 위치 추적 정보
        cv2.putText(frame, "Position Tracking: LEFT | CENTER | RIGHT", (10, height - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
            
        # 하단 정보
        cv2.putText(frame, "Ninja Master - 3 Gesture + Position System", (10, height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def run(self):
        """메인 루프"""
        logger.info("Starting Ninja Master - Simple 3 Gesture System with Position Tracking...")
        logger.info("Gestures: FLICK (표창), FIST (방어), PALM (진동파) + Position (L/C/R)")
        
        try:
            while True:
                success, frame_from_camera = self.cap.read()
                if not success:
                    logger.error("웹캠에서 프레임을 읽을 수 없습니다.")
                    break
                
                current_frame_flipped = cv2.flip(frame_from_camera, 1)
                processed_display_frame, current_debug_messages = self.gesture_recognizer.process_frame(current_frame_flipped)
                current_fps = self.calculate_fps()
                
                if self.debug_mode:
                    self.draw_debug_info(processed_display_frame, current_fps, current_debug_messages)
                
                cv2.imshow("Ninja Master", processed_display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("종료합니다.")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        logger.info("리소스 정리 중...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup()
        logger.info("프로그램 종료.")


def test_mode():
    """OSC 통신 테스트 모드 (위치 정보 포함)"""
    try:
        from test_osc_communication import OSCTester
        
        logger.info("=== OSC 테스트 모드 (3 Gestures + Position) ===")
        tester = OSCTester()
        tester.start_server()

        while True:
            print("\n테스트 옵션:")
            print("1. 3가지 제스처 테스트")
            print("2. 방향별 플릭 테스트")
            print("3. 위치별 제스처 테스트 (NEW)")
            print("4. 개별 제스처 테스트")
            print("5. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                print("\n3가지 핵심 제스처 테스트:")
                gestures = [
                    ("flick", "throw_shuriken", 0.85),
                    ("fist", "block_attack", 0.8),
                    ("palm_push", "shock_wave", 0.9)
                ]
                
                for gesture, action, confidence in gestures:
                    print(f"\n- {gesture.upper()} ({action})")
                    tester.client.send_message("/ninja/gesture/type", gesture)
                    tester.client.send_message("/ninja/gesture/action", action)
                    tester.client.send_message("/ninja/gesture/confidence", confidence)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    
                    if gesture == "flick":
                        tester.client.send_message("/ninja/gesture/direction", [1.0, 0.0])
                        tester.client.send_message("/ninja/gesture/speed", 250.0)
                    
                    time.sleep(1.5)
                    
            elif choice == "2":
                print("\n방향별 플릭 테스트:")
                directions = [
                    ([1.0, 0.0], "오른쪽"),
                    ([-1.0, 0.0], "왼쪽"),
                    ([0.0, 1.0], "아래"),
                    ([0.0, -1.0], "위")
                ]
                for direction, name in directions:
                    print(f"- {name} 플릭")
                    tester.client.send_message("/ninja/gesture/type", "flick")
                    tester.client.send_message("/ninja/gesture/action", "throw_shuriken")
                    tester.client.send_message("/ninja/gesture/direction", direction)
                    tester.client.send_message("/ninja/gesture/speed", 300.0)
                    tester.client.send_message("/ninja/gesture/confidence", 0.9)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    time.sleep(1)
                    
            elif choice == "3":
                print("\n위치별 제스처 테스트:")
                positions = ["left", "center", "right"]
                
                # Flick 위치별 테스트
                print("\n- FLICK 위치별 테스트")
                for position in positions:
                    print(f"  {position.upper()} 플릭")
                    tester.client.send_message("/ninja/gesture/type", "flick")
                    tester.client.send_message("/ninja/gesture/action", "throw_shuriken")
                    tester.client.send_message("/ninja/gesture/position", position)
                    tester.client.send_message("/ninja/gesture/position_action", f"throw_shuriken_{position}")
                    tester.client.send_message("/ninja/gesture/direction", [1.0, 0.0])
                    tester.client.send_message("/ninja/gesture/speed", 300.0)
                    tester.client.send_message("/ninja/gesture/confidence", 0.9)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    time.sleep(1)
                
                # Palm Push 위치별 테스트
                print("\n- PALM PUSH 위치별 테스트")
                for position in positions:
                    print(f"  {position.upper()} 진동파")
                    tester.client.send_message("/ninja/gesture/type", "palm_push")
                    tester.client.send_message("/ninja/gesture/action", "shock_wave")
                    tester.client.send_message("/ninja/gesture/position", position)
                    tester.client.send_message("/ninja/gesture/position_action", f"shock_wave_{position}")
                    tester.client.send_message("/ninja/gesture/confidence", 0.9)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    time.sleep(1)
                    
            elif choice == "4":
                print("\n개별 제스처 선택:")
                print("1. FLICK (표창 던지기)")
                print("2. FIST (공격 막기)")
                print("3. PALM PUSH (진동파)")
                
                gesture_choice = input("선택: ")
                
                if gesture_choice == "1":
                    position = input("위치 선택 (left/center/right): ").lower()
                    if position not in ["left", "center", "right"]:
                        position = "center"
                    
                    tester.client.send_message("/ninja/gesture/type", "flick")
                    tester.client.send_message("/ninja/gesture/action", "throw_shuriken")
                    tester.client.send_message("/ninja/gesture/position", position)
                    tester.client.send_message("/ninja/gesture/position_action", f"throw_shuriken_{position}")
                    tester.client.send_message("/ninja/gesture/direction", [1.0, 0.0])
                    tester.client.send_message("/ninja/gesture/speed", 350.0)
                    tester.client.send_message("/ninja/gesture/confidence", 0.9)
                elif gesture_choice == "2":
                    tester.client.send_message("/ninja/gesture/type", "fist")
                    tester.client.send_message("/ninja/gesture/action", "block_attack")
                    tester.client.send_message("/ninja/gesture/confidence", 0.85)
                elif gesture_choice == "3":
                    position = input("위치 선택 (left/center/right): ").lower()
                    if position not in ["left", "center", "right"]:
                        position = "center"
                        
                    tester.client.send_message("/ninja/gesture/type", "palm_push")
                    tester.client.send_message("/ninja/gesture/action", "shock_wave")
                    tester.client.send_message("/ninja/gesture/position", position)
                    tester.client.send_message("/ninja/gesture/position_action", f"shock_wave_{position}")
                    tester.client.send_message("/ninja/gesture/confidence", 0.95)
                
                tester.client.send_message("/ninja/gesture/hand", "Right")
                print("제스처 전송 완료!")
                
            elif choice == "5":
                break
        
        tester.stop_server()
        logger.info("테스트 모드 종료.")

    except ImportError:
        logger.error("test_osc_communication.py를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"테스트 모드 오류: {e}")


if __name__ == "__main__":
    import sys
    
    is_test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    if is_test_mode:
        test_mode()
    else:
        # 3 제스처 시스템 + 위치 트래킹에 최적화된 설정
        custom_stabilizer_settings = {
            "stability_window": 0.15,      # 빠른 반응
            "confidence_threshold": 0.5,   # 적절한 신뢰도
            "cooldown_time": 0.3          # 짧은 쿨다운
        }
        
        print("\n")
        print("=" * 60)
        print("    닌자 마스터 - 3 제스처 + 위치 트래킹 시스템")
        print("=" * 60)
        print("\n핵심 제스처:")
        print("  1. FLICK     - 손가락 튕기기 → 표창 던지기 (L/C/R)")
        print("  2. FIST      - 주먹 쥐기     → 공격 막기")
        print("  3. PALM PUSH - 5손가락 펴기  → 진동파 (L/C/R)")
        print("\n위치 구분:")
        print("  • LEFT   - 화면 왼쪽 40% 영역")
        print("  • CENTER - 화면 중앙 20% 영역")
        print("  • RIGHT  - 화면 오른쪽 40% 영역")
        print("\n조작법:")
        print("  • Q - 종료")
        print("  • D - 디버그 모드 전환")
        print("=" * 60)
        print("\n")
        
        try:
            tracker = NinjaMasterHandTracker(
                stabilizer_settings_override=custom_stabilizer_settings
            )
            tracker.run()
        except IOError as e:
            logger.critical(f"시작 실패: {e}")
        except KeyboardInterrupt:
            logger.info("\n사용자 중단.")
        except Exception as e:
            logger.critical(f"치명적 오류: {e}")
            import traceback
            traceback.print_exc()