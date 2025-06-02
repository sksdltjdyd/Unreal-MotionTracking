# gesture_recognizer.py - 닌자 게임 인식 시스템 (게임플레이 최적화 버전)

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
    """제스처 타입 정의"""
    NONE = "none"
    FLICK = "flick"
    FIST = "fist"
    PALM_PUSH = "palm_push"
    CROSS_BLOCK = "cross_block"
    HAND_WAVE = "hand_wave"  # 새로운 제스처: 손 흔들기(회피)


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
    
    # 제스처 임계값
    FLICK_SPEED_THRESHOLD = 150
    FIST_ANGLE_THRESHOLD = 100
    PALM_EXTEND_THRESHOLD = 140     # 5손가락 모두 펴기
    WAVE_SPEED_THRESHOLD = 100      # 손 흔들기 속도
    WAVE_DIRECTION_CHANGE = 3       # 방향 전환 횟수
    CROSS_ANGLE_THRESHOLD = 45      # X자 각도 허용 범위
    
    # 안정화 설정
    DEFAULT_STABILITY_WINDOW = 0.25
    DEFAULT_CONFIDENCE_THRESHOLD = 0.4
    DEFAULT_COOLDOWN_TIME = 0.5
    
    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 5
    FPS_BUFFER_SIZE = 30
    
    # 노이즈 필터링
    MOVEMENT_THRESHOLD = 5
    GESTURE_CHANGE_THRESHOLD = 0.3


class EnhancedStabilizer:
    """강화된 제스처 안정화 클래스"""
    def __init__(self, **kwargs):
        self.stability_window = kwargs.get('stability_window', Config.DEFAULT_STABILITY_WINDOW)
        self.confidence_threshold = kwargs.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        self.cooldown_time = kwargs.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
        self.last_gesture_time = {}
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer = deque(maxlen=10)
        self.gesture_confidence_buffer = deque(maxlen=5)
        self.last_sent_gesture = "none"
        self.gesture_change_cooldown = 0
    
    def should_send_gesture(self, gesture_type, confidence, hand_label):
        current_time = time.time()
        
        if gesture_type == "none":
            return False, None
        
        # 양손 제스처는 특별 처리
        if gesture_type == "cross_block" and hand_label == "Both":
            # 크로스 블록은 즉시 전송
            if current_time - self.last_gesture_time.get("cross_block_Both", 0) > 0.5:
                self.last_gesture_time["cross_block_Both"] = current_time
                return True, {"confidence": confidence}
            return False, None
        
        if current_time < self.gesture_change_cooldown:
            if gesture_type != self.last_sent_gesture:
                return False, None
        
        self.gesture_confidence_buffer.append(confidence)
        avg_confidence = np.mean(self.gesture_confidence_buffer)
        
        if avg_confidence < self.confidence_threshold:
            return False, None
        
        self.gesture_buffer.append(gesture_type)
        
        if len(self.gesture_buffer) >= 5:
            gesture_count = {}
            for g in self.gesture_buffer:
                gesture_count[g] = gesture_count.get(g, 0) + 1
            
            most_common = max(gesture_count, key=gesture_count.get)
            if gesture_count[most_common] >= len(self.gesture_buffer) * 0.7:
                gesture_type = most_common
            else:
                return False, None
        
        if gesture_type != self.current_gesture:
            self.current_gesture = gesture_type
            self.current_gesture_start = current_time
            return False, None
        
        if current_time - self.current_gesture_start < self.stability_window:
            return False, None
        
        gesture_key = f"{gesture_type}_{hand_label}"
        if gesture_key in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_key] < self.cooldown_time:
                return False, None
        
        self.last_gesture_time[gesture_key] = current_time
        self.gesture_buffer.clear()
        self.gesture_confidence_buffer.clear()
        self.last_sent_gesture = gesture_type
        self.gesture_change_cooldown = current_time + Config.GESTURE_CHANGE_THRESHOLD
        
        return True, {"confidence": avg_confidence}
    
    def get_statistics(self):
        return {
            "current_gesture": self.current_gesture,
            "stability_progress": time.time() - self.current_gesture_start if self.current_gesture != "none" else 0,
            "buffer_consistency": len(set(self.gesture_buffer)) if self.gesture_buffer else 0
        }
    
    def reset_if_idle(self):
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer.clear()
        self.gesture_confidence_buffer.clear()


class NinjaGestureRecognizer:
    """닌자 게임 제스처 인식기"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        # OSC 설정
        self.client = udp_client.SimpleUDPClient(
            osc_ip or Config.OSC_IP, 
            osc_port or Config.OSC_PORT
        )

        # 안정화 모듈 설정
        if stabilizer_settings is None:
            stabilizer_settings = {}
        
        self.stabilizer = EnhancedStabilizer(**stabilizer_settings)
        logger.info(f"Enhanced Stabilizer initialized with settings: {stabilizer_settings}")

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
        self.position_history = {"Left": deque(maxlen=10), "Right": deque(maxlen=10)}
        self.smoothed_landmarks = {"Left": None, "Right": None}
        
        # 손 흔들기 추적용
        self.wave_positions = {"Left": deque(maxlen=20), "Right": deque(maxlen=20)}
        self.wave_direction_changes = {"Left": 0, "Right": 0}
        self.last_wave_direction = {"Left": 0, "Right": 0}
        
        # 양손 제스처 추적
        self.both_hands_detected_time = 0
        self.cross_block_active = False
        
        # 웹캠 각도 보정을 위한 변수
        self.angle_correction_matrix = self._create_angle_correction_matrix()

        logger.info(f"Ninja Gesture Recognizer initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

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

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지"""
        angles = self.calculate_finger_angles(landmarks)
        
        bent_fingers = 0
        total_fingers = 0
        
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                total_fingers += 1
                if angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                    bent_fingers += 1
        
        if total_fingers > 0 and bent_fingers >= total_fingers * 0.6:
            confidence = bent_fingers / total_fingers
            return True, max(0.6, confidence)
        
        return False, 0.0

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0 or dt > 0.5:
            return False, None, 0.0
        
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP]
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
                    
                    angle = np.arctan2(direction_normalized[1], direction_normalized[0])
                    angle_deg = np.degrees(angle)
                    
                    if -45 <= angle_deg <= 45:
                        simplified_direction = [1.0, 0.0]
                    elif angle_deg > 45 and angle_deg <= 135:
                        simplified_direction = [0.0, 1.0]
                    elif angle_deg < -45 and angle_deg >= -135:
                        simplified_direction = [0.0, -1.0]
                    else:
                        simplified_direction = [-1.0, 0.0]
                    
                    best_flick = simplified_direction
                    best_velocity = velocity
        
        if best_flick:
            confidence = min(0.7 + (best_velocity - Config.FLICK_SPEED_THRESHOLD) / 500, 1.0)
            return True, best_flick, best_velocity
        
        return False, None, 0.0

    def detect_palm_push(self, landmarks, hand_label):
        """손바닥 밀기 감지 - 5손가락 모두 펴기"""
        finger_angles = self.calculate_finger_angles(landmarks)
        
        # 모든 손가락이 펴져있는지 확인
        extended_fingers = 0
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in finger_angles:
                # 엄지는 다른 기준 적용
                threshold = 100 if finger == 'thumb' else Config.PALM_EXTEND_THRESHOLD
                if finger_angles[finger] > threshold:
                    extended_fingers += 1
        
        # 5개 손가락 모두 펴져있으면 palm push
        if extended_fingers >= 5:
            # z축 체크로 추가 확인
            wrist_z = landmarks[self.WRIST].z
            middle_mcp_z = landmarks[9].z
            
            z_diff = middle_mcp_z - wrist_z
            confidence = min(0.8 + abs(z_diff), 1.0)
            return True, confidence
        
        return False, 0.0

    def detect_hand_wave(self, current_landmarks, hand_label, img_width, img_height):
        """손 흔들기(회피) 감지"""
        # 손목 위치 추적
        wrist = current_landmarks[self.WRIST]
        wrist_x, wrist_y, _ = self._correct_landmark_position(wrist)
        current_pos = np.array([wrist_x * img_width, wrist_y * img_height])
        
        self.wave_positions[hand_label].append(current_pos)
        
        if len(self.wave_positions[hand_label]) < 10:
            return False, 0.0
        
        # 최근 위치들의 x좌표 변화 분석
        positions = np.array(self.wave_positions[hand_label])
        x_positions = positions[:, 0]
        
        # 방향 변화 감지
        direction_changes = 0
        last_direction = 0
        
        for i in range(1, len(x_positions)):
            diff = x_positions[i] - x_positions[i-1]
            if abs(diff) > Config.MOVEMENT_THRESHOLD:
                current_direction = 1 if diff > 0 else -1
                if last_direction != 0 and current_direction != last_direction:
                    direction_changes += 1
                last_direction = current_direction
        
        # 좌우로 충분히 흔들었는지 확인
        if direction_changes >= Config.WAVE_DIRECTION_CHANGE:
            # 속도 체크
            total_distance = np.sum(np.abs(np.diff(x_positions)))
            avg_speed = total_distance / len(x_positions)
            
            if avg_speed > Config.WAVE_SPEED_THRESHOLD / 10:  # 평균 속도
                self.wave_positions[hand_label].clear()
                confidence = min(0.7 + direction_changes * 0.1, 1.0)
                return True, confidence
        
        # 버퍼가 꽉 차면 오래된 데이터 제거
        if len(self.wave_positions[hand_label]) >= 20:
            for _ in range(5):
                self.wave_positions[hand_label].popleft()
        
        return False, 0.0

    def detect_cross_block(self, left_landmarks, right_landmarks):
        """양손 X자 블록 감지"""
        # 양손의 손목과 팔꿈치 위치 추정
        left_wrist = np.array([left_landmarks[self.WRIST].x, left_landmarks[self.WRIST].y])
        right_wrist = np.array([right_landmarks[self.WRIST].x, right_landmarks[self.WRIST].y])
        
        # 손목 위치가 교차하는지 확인 (X자 형태)
        # 왼손이 오른쪽에, 오른손이 왼쪽에 있어야 함
        if left_wrist[0] > 0.5 and right_wrist[0] < 0.5:
            # 손목 높이가 비슷한지 확인
            height_diff = abs(left_wrist[1] - right_wrist[1])
            if height_diff < 0.2:  # 20% 이내
                # 두 손목 사이의 거리가 적당한지 확인
                wrist_distance = self.calculate_distance(left_wrist, right_wrist)
                if 0.1 < wrist_distance < 0.4:
                    # 손이 펴져있는지 확인 (주먹이 아닌지)
                    left_angles = self.calculate_finger_angles(left_landmarks)
                    right_angles = self.calculate_finger_angles(right_landmarks)
                    
                    left_extended = sum(1 for a in left_angles.values() if a > 120) >= 3
                    right_extended = sum(1 for a in right_angles.values() if a > 120) >= 3
                    
                    if left_extended and right_extended:
                        confidence = 0.9
                        return True, confidence
        
        return False, 0.0

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape, both_hands_data=None):
        """통합 제스처 인식"""
        landmarks = self._smooth_landmarks(hand_landmarks_obj.landmark, hand_label)
        height, width = img_shape[:2]
        
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # 양손 제스처 체크 (우선순위 최상)
        if both_hands_data and "Left" in both_hands_data and "Right" in both_hands_data:
            is_cross, cross_conf = self.detect_cross_block(
                both_hands_data["Left"], 
                both_hands_data["Right"]
            )
            if is_cross:
                return GestureType.CROSS_BLOCK, {"confidence": cross_conf, "hands": "both"}

        # 단일 손 제스처
        # 1. 플릭
        is_flick, flick_dir, flick_speed = self.detect_flick(landmarks, hand_label, width, height)
        if is_flick:
            current_gesture = GestureType.FLICK
            gesture_data = {"direction": flick_dir, "speed": flick_speed, "confidence": 0.8}
        else:
            # 2. 손 흔들기 (회피)
            is_wave, wave_conf = self.detect_hand_wave(landmarks, hand_label, width, height)
            if is_wave:
                current_gesture = GestureType.HAND_WAVE
                gesture_data = {"confidence": wave_conf}
            else:
                # 3. 손바닥 밀기 (5손가락)
                is_push, push_conf = self.detect_palm_push(landmarks, hand_label)
                if is_push:
                    current_gesture = GestureType.PALM_PUSH
                    gesture_data = {"confidence": push_conf}
                else:
                    # 4. 주먹
                    is_fist, fist_conf = self.detect_fist(landmarks)
                    if is_fist:
                        current_gesture = GestureType.FIST
                        gesture_data = {"confidence": fist_conf}

        self.prev_landmarks[hand_label] = hand_landmarks_obj.landmark
        return current_gesture, gesture_data

    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type")
            msg_builder.add_arg(gesture_type_str)
            bundle_builder.add_content(msg_builder.build())
            
            msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence")
            msg_builder.add_arg(float(confidence))
            bundle_builder.add_content(msg_builder.build())

            if "direction" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                direction_val = gesture_data["direction"]
                if isinstance(direction_val, list) and len(direction_val) == 2:
                    msg_builder.add_arg(float(direction_val[0]))
                    msg_builder.add_arg(float(direction_val[1]))
                else:
                    msg_builder.add_arg(str(direction_val))
                bundle_builder.add_content(msg_builder.build())

            if "speed" in gesture_data:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                msg_builder.add_arg(float(gesture_data["speed"]))
                bundle_builder.add_content(msg_builder.build())
            
            # 양손 제스처인 경우
            if gesture_data.get("hands") == "both":
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
                msg_builder.add_arg("Both")
                bundle_builder.add_content(msg_builder.build())
            else:
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
                msg_builder.add_arg(hand_label)
                bundle_builder.add_content(msg_builder.build())
            
            self.client.send(bundle_builder.build())
            logger.info(f"OSC Sent: {gesture_type_str} ({hand_label}), Conf: {confidence:.2f}")

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
                # 양손 데이터 수집
                both_hands_data = {}
                
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label
                    
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 양손 제스처를 위해 랜드마크 저장
                    both_hands_data[hand_label] = hand_landmarks_obj.landmark
                
                # 양손이 모두 감지되면 크로스 블록 체크
                if len(both_hands_data) == 2:
                    is_cross, cross_conf = self.detect_cross_block(
                        both_hands_data.get("Left"),
                        both_hands_data.get("Right")
                    )
                    if is_cross:
                        should_send, _ = self.stabilizer.should_send_gesture(
                            "cross_block", cross_conf, "Both"
                        )
                        if should_send:
                            self.send_gesture_osc(
                                GestureType.CROSS_BLOCK,
                                {"confidence": cross_conf, "hands": "both"},
                                "Both"
                            )
                            debug_messages_for_frame.append("Both hands: cross_block ✓")
                
                # 개별 손 제스처 처리
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label
                    
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape, both_hands_data
                    )
                    
                    # 크로스 블록이 아닌 경우만 처리
                    if raw_gesture_type != GestureType.CROSS_BLOCK:
                        should_send, stabilized_gesture_data = self.stabilizer.should_send_gesture(
                            raw_gesture_type.value,
                            raw_gesture_data.get("confidence", 0.0),
                            hand_label
                        )
                        
                        if should_send and raw_gesture_type != GestureType.NONE:
                            self.send_gesture_osc(raw_gesture_type, raw_gesture_data, hand_label)
                            debug_messages_for_frame.append(f"{hand_label}: {raw_gesture_type.value} ✓")
                        else:
                            stabilizer_stats = self.stabilizer.get_statistics()
                            pending_gesture = stabilizer_stats.get("current_gesture", "none")
                            
                            if pending_gesture != "none":
                                progress = stabilizer_stats.get("stability_progress", 0)
                                progress_percent = min(progress / self.stabilizer.stability_window * 100, 100)
                                consistency = stabilizer_stats.get("buffer_consistency", 0)
                                debug_messages_for_frame.append(
                                    f"{hand_label}: {pending_gesture} ({progress_percent:.0f}% | C:{consistency})"
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
        """디버그 정보 그리기"""
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 제스처 정보
        y_offset = 70
        for message in debug_messages_list:
            color = (0, 255, 255) if "✓" in message else (255, 255, 0)
            cv2.putText(frame, message, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y_offset += 30
        
        # 제스처 가이드
        guide_y = Config.CAMERA_HEIGHT - 200
        cv2.putText(frame, "== Gesture Guide ==", (10, guide_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        gestures = [
            ("Flick", "Quick finger movement - Attack"),
            ("Fist", "Close hand - Power attack"),
            ("Palm Push", "Open all 5 fingers - Force push"),
            ("Hand Wave", "Wave hand left-right - Dodge"),
            ("Cross Block", "Cross both hands - Defense")
        ]
        
        for i, (gesture, desc) in enumerate(gestures):
            y_pos = guide_y + 25 + (i * 20)
            cv2.putText(frame, f"- {gesture}: {desc}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
        # 하단 정보
        cv2.putText(frame, "Ninja Master - Gameplay Optimized", (10, Config.CAMERA_HEIGHT - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, Config.CAMERA_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def run(self):
        """메인 루프"""
        logger.info("Starting Ninja Master - Gameplay Optimized Version...")
        
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
    """OSC 통신 테스트 모드"""
    try:
        from test_osc_communication import OSCTester
        
        logger.info("=== OSC 테스트 모드 ===")
        tester = OSCTester()
        tester.start_server()

        while True:
            print("\n테스트 옵션:")
            print("1. 모든 제스처 테스트")
            print("2. 방향별 플릭 테스트")
            print("3. 양손 제스처 테스트")
            print("4. 손 흔들기 테스트")
            print("5. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                tester.test_all_gestures()
            elif choice == "2":
                print("방향별 플릭 전송:")
                directions = [
                    ([1.0, 0.0], "오른쪽"),
                    ([-1.0, 0.0], "왼쪽"),
                    ([0.0, 1.0], "아래"),
                    ([0.0, -1.0], "위")
                ]
                for direction, name in directions:
                    print(f"- {name} 플릭")
                    tester.client.send_message("/ninja/gesture/type", "flick")
                    tester.client.send_message("/ninja/gesture/direction", direction)
                    tester.client.send_message("/ninja/gesture/confidence", 0.9)
                    time.sleep(1)
            elif choice == "3":
                print("양손 제스처 테스트:")
                print("- Cross Block")
                tester.client.send_message("/ninja/gesture/type", "cross_block")
                tester.client.send_message("/ninja/gesture/hand", "Both")
                tester.client.send_message("/ninja/gesture/confidence", 0.9)
            elif choice == "4":
                print("손 흔들기 테스트:")
                tester.client.send_message("/ninja/gesture/type", "hand_wave")
                tester.client.send_message("/ninja/gesture/hand", "Right")
                tester.client.send_message("/ninja/gesture/confidence", 0.8)
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
        # 게임플레이에 최적화된 안정화 설정
        custom_stabilizer_settings = {
            "stability_window": 0.2,      # 적절한 반응 속도
            "confidence_threshold": 0.45,  # 중간 신뢰도
            "cooldown_time": 0.4          # 연속 입력 방지
        }
        
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