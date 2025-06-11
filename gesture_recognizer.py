# gesture_recognizer.py - 닌자 게임 인식 시스템 (Palm Push 개선 및 오류 수정)

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pythonosc import udp_client, osc_bundle_builder, osc_message_builder
from enum import Enum
import logging
import sys
import traceback

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
    FLICK_SPEED_THRESHOLD = 180      # 속도 임계값
    FIST_ANGLE_THRESHOLD = 110       # 주먹 인식
    PALM_EXTEND_THRESHOLD = 130      # 5손가락 펴기 인식

    # Flick 정확도 - 검지와 중지 거리 (가까워야 함)
    FLICK_FINGER_DISTANCE_THRESHOLD = 0.03  # 화면 대비 3% 이내

    # Palm Push 정확도 - 검지와 중지 거리 (멀어야 함)
    PALM_FINGER_DISTANCE_THRESHOLD = 0.06  # 화면 대비 6% 이상

    # 안정화 설정 - 정확한 인식을 위해 강화
    DEFAULT_STABILITY_WINDOW = 0.4      # 0.4초 동안 제스처 유지 필요
    DEFAULT_CONFIDENCE_THRESHOLD = 0.75 # 75% 이상 신뢰도 필요
    DEFAULT_COOLDOWN_TIME = 0.8         # 0.8초 쿨다운으로 연속 발동 방지

    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30

    # 노이즈 필터링 - 더 엄격하게
    MOVEMENT_THRESHOLD = 10             # 10픽셀 이상 움직여야 인식
    GESTURE_CHANGE_THRESHOLD = 0.5      # 50% 이상 변화해야 제스처 변경
    POSITION_CHANGE_THRESHOLD = 0.15    # 위치 변경 임계값

    # 위치 트래킹 설정
    POSITION_LEFT_THRESHOLD = 0.33      # 화면의 33% 이하는 좌측
    POSITION_RIGHT_THRESHOLD = 0.66     # 화면의 66% 이상은 우측
    POSITION_TRACKING_SMOOTHING = 0.8   # 위치 스무딩 계수


class SimpleStabilizer:
    """단순화된 제스처 안정화"""
    def __init__(self, **kwargs):
        self.stability_window = kwargs.get('stability_window', Config.DEFAULT_STABILITY_WINDOW)
        self.confidence_threshold = kwargs.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        self.cooldown_time = kwargs.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
        self.last_gesture_time = {}
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer = deque(maxlen=10)  # 버퍼 크기 증가
        self.position_buffer = deque(maxlen=5)  # 위치 버퍼
        self.last_valid_position = "center"
        self.last_sent_gesture = "none"

    def should_send_gesture(self, gesture_type, confidence, hand_label, position=None):
        current_time = time.time()

        if gesture_type == "none":
            return False, None

        # 신뢰도 체크 - 더 엄격하게
        if confidence < self.confidence_threshold:
            return False, None

        # 위치 안정화
        if position:
            self.position_buffer.append(position)
            if len(self.position_buffer) >= 3:
                position_counts = {}
                for pos in self.position_buffer:
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                most_common_position = max(position_counts, key=position_counts.get)

                if position_counts[most_common_position] >= len(self.position_buffer) * 0.6:
                    self.last_valid_position = most_common_position
                position = self.last_valid_position

        self.gesture_buffer.append(gesture_type)

        # 버퍼의 70% 이상이 같은 제스처면 인식
        if len(self.gesture_buffer) >= 7:
            gesture_count = {}
            for g in self.gesture_buffer:
                gesture_count[g] = gesture_count.get(g, 0) + 1

            most_common = max(gesture_count, key=gesture_count.get)
            if gesture_count[most_common] >= len(self.gesture_buffer) * 0.7:
                gesture_type = most_common
            else:
                return False, None
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
    """닌자 게임 제스처 인식기 - 개선된 Palm Push"""

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

        # 위치 추적용 변수
        self.hand_positions = {"Left": "center", "Right": "center"}
        self.smoothed_hand_x = {"Left": 0.5, "Right": 0.5}

        # 웹캠 각도 보정을 위한 변수
        self.angle_correction_matrix = self._create_angle_correction_matrix()

        logger.info(f"Ninja Gesture Recognizer (Improved Palm Push) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

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
            'thumb': [1, 2, 4],
            'index': [5, 6, 8],
            'middle': [9, 10, 12],
            'ring': [13, 14, 16],
            'pinky': [17, 18, 20]
        }
        
        for finger, joints in finger_joints_indices.items():
            p1 = np.array([landmarks[joints[0]].x, landmarks[joints[0]].y])
            p2 = np.array([landmarks[joints[1]].x, landmarks[joints[1]].y])
            p3 = np.array([landmarks[joints[2]].x, landmarks[joints[2]].y])
            
            angles[finger] = self.calculate_angle(p1, p2, p3)
        return angles

    def calculate_hand_position(self, landmarks, hand_label):
        """손의 화면상 위치(좌/중앙/우) 판단"""
        wrist_x = landmarks[self.WRIST].x
        middle_mcp_x = landmarks[9].x  # 중지 MCP
        hand_center_x = (wrist_x + middle_mcp_x) / 2

        self.smoothed_hand_x[hand_label] = (
            Config.POSITION_TRACKING_SMOOTHING * self.smoothed_hand_x[hand_label] +
            (1 - Config.POSITION_TRACKING_SMOOTHING) * hand_center_x
        )
        
        smoothed_x = self.smoothed_hand_x[hand_label]

        if smoothed_x < Config.POSITION_LEFT_THRESHOLD:
            position = "left"
        elif smoothed_x > Config.POSITION_RIGHT_THRESHOLD:
            position = "right"
        else:
            position = "center"
            
        self.hand_positions[hand_label] = position
        return position, smoothed_x

    def check_finger_tips_distance(self, landmarks, gesture_type="flick"):
        """검지와 중지 끝점 사이의 거리 체크 (Flick / Palm Push 구분)"""
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        distance = self.calculate_distance(
            [index_tip.x, index_tip.y],
            [middle_tip.x, middle_tip.y]
        )

        if gesture_type == "flick":
            return distance < Config.FLICK_FINGER_DISTANCE_THRESHOLD, distance
        else:  # palm_push
            return distance > Config.PALM_FINGER_DISTANCE_THRESHOLD, distance

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지 - 공격 막기"""
        angles = self.calculate_finger_angles(landmarks)
        bent_fingers = 0
        total_fingers = 0
        
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                total_fingers += 1
                if angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                    bent_fingers += 1
        
        if total_fingers > 0 and bent_fingers >= 3:
            confidence = 0.7 + (bent_fingers / total_fingers) * 0.3
            return True, confidence
        
        return False, 0.0

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지 - 표창 던지기 (검지-중지 붙어있어야 함)"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0, "center"

        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0 or dt > 0.5:
            return False, None, 0.0, "center"

        position, _ = self.calculate_hand_position(current_landmarks, hand_label)
        
        # ★ 검지와 중지 끝이 붙어있는지 확인
        fingers_together, finger_distance = self.check_finger_tips_distance(current_landmarks, "flick")
        if not fingers_together:
            logger.debug(f"Flick failed - fingers not close: distance={finger_distance:.3f}")
            return False, None, 0.0, position

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

            if distance < Config.MOVEMENT_THRESHOLD * 2:
                continue
            
            velocity = distance / dt

            if velocity > Config.FLICK_SPEED_THRESHOLD and velocity > best_velocity:
                simplified_direction = [1.0, 0.0]
                
                finger_angles = self.calculate_finger_angles(current_landmarks)
                finger_name = 'index' if finger_tip == self.INDEX_TIP else 'middle'

                if finger_name in finger_angles and finger_angles[finger_name] > 120:
                    best_flick = simplified_direction
                    best_velocity = velocity

        if best_flick:
            distance_factor = 1.0 - (finger_distance / Config.FLICK_FINGER_DISTANCE_THRESHOLD)
            confidence = min(0.7 + (best_velocity - Config.FLICK_SPEED_THRESHOLD) / 500 + distance_factor * 0.1, 1.0)
            logger.info(f"Flick detected! Fingers distance: {finger_distance:.3f}, Velocity: {best_velocity:.1f}")
            return True, best_flick, best_velocity, position
        
        return False, None, 0.0, position

    def detect_palm_push(self, landmarks, hand_label):
        """손바닥 밀기 감지 - 진동파 (검지-중지 떨어져 있어야 함)"""
        position, _ = self.calculate_hand_position(landmarks, hand_label)
        
        # ★ 수정된 조건: 검지와 중지가 충분히 떨어져 있는지 확인
        fingers_apart, finger_distance = self.check_finger_tips_distance(landmarks, "palm_push")
        if not fingers_apart:
            logger.debug(f"Palm Push failed - fingers too close: distance={finger_distance:.3f}")
            return False, 0.0, position

        finger_angles = self.calculate_finger_angles(landmarks)
        extended_fingers = 0
        total_fingers = 0

        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in finger_angles:
                total_fingers += 1
                threshold = 150 if finger == 'thumb' else Config.PALM_EXTEND_THRESHOLD
                if finger_angles[finger] > threshold:
                    extended_fingers += 1
        
        if total_fingers >= 5 and extended_fingers >= 5:
            wrist_z = landmarks[self.WRIST].z
            middle_mcp_z = landmarks[9].z
            z_diff = middle_mcp_z - wrist_z
            base_confidence = 0.8
            distance_factor = min((finger_distance - Config.PALM_FINGER_DISTANCE_THRESHOLD) / 0.05, 1.0)

            if z_diff < 0:  # 손바닥이 카메라를 향함
                confidence = min(base_confidence + abs(z_diff) * 0.5 + distance_factor * 0.1, 1.0)
            else:
                confidence = min(base_confidence + distance_factor * 0.1, 1.0)
            
            logger.info(f"Palm Push detected! Finger distance: {finger_distance:.3f}")
            return True, confidence, position

        return False, 0.0, position

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        """통합 제스처 인식 - 위치 정보 포함"""
        landmarks = self._smooth_landmarks(hand_landmarks_obj.landmark, hand_label)
        height, width = img_shape[:2]
        
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # 우선순위에 따른 제스처 인식
        # 1. 플릭 (표창 던지기)
        is_flick, flick_dir, flick_speed, flick_position = self.detect_flick(
            landmarks, hand_label, width, height
        )
        if is_flick:
            current_gesture = GestureType.FLICK
            gesture_data = {
                "direction": flick_dir,
                "speed": flick_speed,
                "confidence": 0.85, # 임시 신뢰도
                "action": "throw_shuriken",
                "position": flick_position
            }
        else:
            # 2. 주먹 (공격 막기)
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
                        "position": push_position
                    }

        self.prev_landmarks[hand_label] = hand_landmarks_obj.landmark
        return current_gesture, gesture_data

    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value
            action = gesture_data.get("action", "")
            position = gesture_data.get("position", "center")

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            # 기본 정보
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type")
            msg_builder.add_arg(gesture_type_str)
            bundle_builder.add_content(msg_builder.build())

            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/action")
            msg_builder.add_arg(action)
            bundle_builder.add_content(msg_builder.build())
            
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence")
            msg_builder.add_arg(float(confidence))
            bundle_builder.add_content(msg_builder.build())

            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
            msg_builder.add_arg(hand_label)
            bundle_builder.add_content(msg_builder.build())

            # 위치 정보 (flick과 palm_push에만 해당)
            if gesture_type_str in ["flick", "palm_push"] and "position" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position")
                msg_builder.add_arg(position)
                bundle_builder.add_content(msg_builder.build())
                
                position_action = f"{action}_{position}"
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position_action")
                msg_builder.add_arg(position_action)
                bundle_builder.add_content(msg_builder.build())

            # 방향 및 속도 정보 (플릭용)
            if "direction" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                direction_val = gesture_data["direction"]
                if isinstance(direction_val, list) and len(direction_val) == 2:
                    msg_builder.add_arg(float(direction_val[0]))
                    msg_builder.add_arg(float(direction_val[1]))
                bundle_builder.add_content(msg_builder.build())

            if "speed" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                msg_builder.add_arg(float(gesture_data["speed"]))
                bundle_builder.add_content(msg_builder.build())

            self.client.send(bundle_builder.build())
            
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
        """프레임 처리 및 시각화"""
        frame_to_draw_on = frame_input.copy()
        debug_messages_for_frame = []

        try:
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label
                    
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    landmarks = hand_landmarks_obj.landmark
                    h, w = frame_input.shape[:2]

                    # --- 개선된 시각화 로직 ---
                    index_tip = landmarks[self.INDEX_TIP]
                    middle_tip = landmarks[self.MIDDLE_TIP]
                    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                    middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

                    is_close_for_flick, finger_dist = self.check_finger_tips_distance(landmarks, "flick")
                    is_apart_for_palm, _ = self.check_finger_tips_distance(landmarks, "palm_push")

                    line_color = (0, 0, 255)  # 기본: 빨간색 (조건 불만족)
                    status_text = ""
                    if is_close_for_flick:
                        line_color = (0, 255, 0)  # 초록색: Flick 준비 완료
                        status_text = "FLICK READY"
                    elif is_apart_for_palm:
                        line_color = (0, 255, 255) # 노란색: Palm Push 준비 완료
                        status_text = "PALM READY"
                    
                    cv2.line(frame_to_draw_on, index_pos, middle_pos, line_color, 3)
                    
                    mid_point = ((index_pos[0] + middle_pos[0]) // 2, (index_pos[1] + middle_pos[1]) // 2)
                    cv2.putText(frame_to_draw_on, f"{finger_dist:.3f}", mid_point, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    if status_text:
                        cv2.putText(frame_to_draw_on, status_text, (mid_point[0] - 40, mid_point[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)

                    # --- 제스처 인식 및 전송 ---
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape
                    )
                    
                    position_for_stabilizer = raw_gesture_data.get("position")
                    should_send, _ = self.stabilizer.should_send_gesture(
                        raw_gesture_type.value,
                        raw_gesture_data.get("confidence", 0.0),
                        hand_label,
                        position_for_stabilizer
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
            traceback.print_exc()
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
        
        logger.info("Ninja Master Hand Tracker (Improved Gestures) initialized.")

    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        time_difference = current_time - self.last_time
        
        if time_difference > 0:
            fps = 1.0 / time_difference
            self.fps_counter.append(fps)
        
        self.last_time = current_time
        return np.mean(self.fps_counter) if self.fps_counter else 0.0

    def draw_debug_info(self, frame, fps, debug_messages_list):
        """디버그 정보 그리기"""
        height, width = frame.shape[:2]
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 화면 3등분 가이드라인
        left_line = int(width * Config.POSITION_LEFT_THRESHOLD)
        right_line = int(width * Config.POSITION_RIGHT_THRESHOLD)
        for y in range(0, height, 20):
            cv2.line(frame, (left_line, y), (left_line, y + 10), (100, 100, 100), 2)
            cv2.line(frame, (right_line, y), (right_line, y + 10), (100, 100, 100), 2)
        
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
        
        # 거리 임계값 표시
        guide_y = height - 180
        cv2.putText(frame, f"Flick Dist < {Config.FLICK_FINGER_DISTANCE_THRESHOLD:.3f} (Green Line)", 
                    (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Palm Dist > {Config.PALM_FINGER_DISTANCE_THRESHOLD:.3f} (Yellow Line)", 
                    (10, guide_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 제스처 가이드
        guide_y += 70
        cv2.putText(frame, "== Gestures ==", (10, guide_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        gestures = [
            ("FLICK", "Index-Mid Close + Fast Move", (0, 255, 0)),
            ("FIST", "Close Hand", (255, 255, 0)),
            ("PALM", "All Fingers Open + Index-Mid Apart", (0, 255, 255))
        ]
        for i, (name, desc, color) in enumerate(gestures):
            y_pos = guide_y + 25 + (i * 25)
            cv2.putText(frame, f"- {name}: {desc}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 하단 정보
        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def run(self):
        """메인 루프"""
        logger.info("Starting Ninja Master - Enhanced Accuracy System...")
        # --- 수정된 로그 메시지 ---
        logger.info("Flick: Index & Middle fingers must be CLOSE")
        logger.info("Palm Push: Index & Middle fingers must be APART")
        logger.info(f"Flick distance threshold: < {Config.FLICK_FINGER_DISTANCE_THRESHOLD:.3f}")
        logger.info(f"Palm distance threshold: > {Config.PALM_FINGER_DISTANCE_THRESHOLD:.3f}") # 오류 수정
        
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
                
                # 창 크기를 화면에 맞게 조절하여 표시
                window_name = "Ninja Master - Enhanced"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
                cv2.imshow(window_name, processed_display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("종료합니다.")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}")
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
    """OSC 통신 테스트 모드"""
    try:
        # 이 테스트를 실행하려면 동일한 폴더에 test_osc_communication.py 파일이 필요합니다.
        from test_osc_communication import OSCTester
        
        logger.info("=== OSC 테스트 모드 (Enhanced Gestures) ===")
        tester = OSCTester()
        tester.start_server()

        while True:
            print("\n테스트 옵션:")
            print("1. 위치별 제스처 테스트")
            print("2. 손가락 거리 조건 설명")
            print("3. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                gestures = [
                    ("flick", "throw_shuriken", 0.85),
                    ("fist", "block_attack", 0.8),
                    ("palm_push", "shock_wave", 0.9)
                ]
                positions = ["left", "center", "right"]
                
                for gesture, action, confidence in gestures:
                    for pos in positions:
                        if gesture == "fist" and pos != "center": continue # 주먹은 위치 없음
                        
                        print(f"\n- {gesture.upper()} @ {pos.upper()}")
                        
                        # --- 수정된 주석 ---
                        if gesture == "flick":
                            print("  (검지와 중지가 붙어있는 상태)")
                        elif gesture == "palm_push":
                            print("  (검지와 중지가 떨어진 상태)")
                        
                        # OSC 메시지 전송
                        tester.client.send_message("/ninja/gesture/type", gesture)
                        tester.client.send_message("/ninja/gesture/action", action)
                        tester.client.send_message("/ninja/gesture/confidence", confidence)
                        tester.client.send_message("/ninja/gesture/hand", "Right")
                        if gesture != "fist":
                            tester.client.send_message("/ninja/gesture/position", pos)
                        
                        time.sleep(1.5)
            
            elif choice == "2":
                print("\n손가락 거리 조건:")
                print(f"Flick 임계값 (가까움): < {Config.FLICK_FINGER_DISTANCE_THRESHOLD:.3f}")
                # --- 오류 수정 ---
                print(f"Palm Push 임계값 (멀음): > {Config.PALM_FINGER_DISTANCE_THRESHOLD:.3f}")
                print("\n실행 화면에서 아래와 같이 표시됩니다:")
                print("- 초록선: Flick 조건 만족 (검지-중지 가까움)")
                print("- 노란선: Palm Push 조건 만족 (검지-중지 멈)")
                print("- 빨간선: 두 조건 모두 불만족")

            elif choice == "3":
                break
        
        tester.stop_server()
        logger.info("테스트 모드 종료.")

    except ImportError:
        logger.error("test_mode를 실행하려면 test_osc_communication.py 파일이 필요합니다.")
    except Exception as e:
        logger.error(f"테스트 모드 오류: {e}")


if __name__ == "__main__":
    is_test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    if is_test_mode:
        test_mode()
    else:
        # 안정적인 설정
        custom_stabilizer_settings = {
            "stability_window": 0.4,
            "confidence_threshold": 0.75,
            "cooldown_time": 0.8
        }
        
        print("\n" + "=" * 60)
        print("      닌자 마스터 - 향상된 정확도 시스템")
        print("=" * 60)
        print("\n핵심 개선사항:")
        # --- 수정된 설명 ---
        print("  • FLICK: '검지'와 '중지'가 가까워야 인식됩니다.")
        print("  • PALM PUSH: '검지'와 '중지'가 멀어야 인식됩니다.")
        print(f"  • Flick 거리 임계값: < {Config.FLICK_FINGER_DISTANCE_THRESHOLD:.3f}")
        print(f"  • Palm 거리 임계값: > {Config.PALM_FINGER_DISTANCE_THRESHOLD:.3f}")
        print("\n화면 표시:")
        print("  • 초록선: Flick 조건 만족 (손가락 가까움)")
        print("  • 노란선: Palm Push 조건 만족 (손가락 멈)")
        print("  • 빨간선: 조건 불만족")
        print("\n조작법:")
        print("  • Q - 종료")
        print("  • D - 디버그 정보 ON/OFF")
        print("=" * 60 + "\n")
        
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
            logger.critical(f"치명적 오류 발생: {e}")
            traceback.print_exc()