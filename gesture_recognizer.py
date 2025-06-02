# gesture_recognizer.py - 닌자 게임 인식 시스템 (심플 3제스처 버전)

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
    FLICK = "flick"      # 표창 던지기
    FIST = "fist"        # 공격 막기
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
    DEFAULT_COOLDOWN_TIME = 0.3       # 짧은 쿨다운
    
    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30
    
    # 노이즈 필터링
    MOVEMENT_THRESHOLD = 3
    GESTURE_CHANGE_THRESHOLD = 0.2
    
    # 위치 트래킹 설정 (새로 추가)
    POSITION_LEFT_THRESHOLD = 0.4     # 화면의 40% 이하는 좌측
    POSITION_RIGHT_THRESHOLD = 0.6    # 화면의 60% 이상은 우측
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
        self.gesture_buffer = deque(maxlen=5) # 버퍼 크기를 좀 더 명시적으로 (혹은 Config에서 관리)
        self.last_sent_gesture = "none"
    
    def should_send_gesture(self, gesture_type, confidence, hand_label):
        current_time = time.time()
        
        if gesture_type == "none": # GestureType.NONE.value 대신 "none" 사용 일관성 유지
            self.gesture_buffer.append("none") # NONE도 버퍼에 추가하여 현재 상태 반영
            # 현재 제스처가 NONE이 아닐 때만 리셋 로직 고려
            if self.current_gesture != "none" and all(g == "none" for g in self.gesture_buffer):
                 self.current_gesture = "none"
                 self.current_gesture_start = 0
            return False, None
        
        # 신뢰도 체크
        if confidence < self.confidence_threshold:
            return False, None
        
        self.gesture_buffer.append(gesture_type)
        
        # 버퍼의 60% 이상이 같은 제스처면 인식 (버퍼가 충분히 찼을때)
        if len(self.gesture_buffer) == self.gesture_buffer.maxlen: # 버퍼가 꽉 찼을 때만 판단
            gesture_count = {}
            for g in self.gesture_buffer:
                gesture_count[g] = gesture_count.get(g, 0) + 1
            
            most_common = max(gesture_count, key=gesture_count.get)
            if gesture_count[most_common] >= len(self.gesture_buffer) * 0.6:
                processed_gesture_type = most_common
            else:
                # 버퍼가 특정 제스처로 충분히 채워지지 않으면 불안정한 상태로 간주
                # self.current_gesture = "none" # 너무 자주 리셋될 수 있음
                # self.current_gesture_start = 0
                return False, None
        else: # 버퍼가 아직 덜 참
            return False, None

        # 새로운 제스처 시작 또는 변경
        if processed_gesture_type != self.current_gesture:
            self.current_gesture = processed_gesture_type
            self.current_gesture_start = current_time
            return False, None # 안정화 시간 대기
        
        # 안정화 시간 체크
        if current_time - self.current_gesture_start < self.stability_window:
            return False, None
        
        # 쿨다운 체크 (동일 제스처에 대한 쿨다운)
        gesture_key = f"{self.current_gesture}_{hand_label}" # current_gesture 사용
        if gesture_key in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_key] < self.cooldown_time:
                return False, None
        
        # 제스처 전송 조건 만족
        self.last_gesture_time[gesture_key] = current_time
        # self.gesture_buffer.clear() # 버퍼는 계속 흘러가도록 두거나, 전송 후 명시적으로 리셋
        self.last_sent_gesture = self.current_gesture 
        
        return True, {"confidence": confidence} # 전송할 때는 원래 confidence 사용
    
    def get_statistics(self):
        progress = 0
        if self.current_gesture != "none" and self.stability_window > 0:
             progress = (time.time() - self.current_gesture_start) / self.stability_window
        return {
            "current_gesture": self.current_gesture,
            "stability_progress": min(progress, 1.0) # 0과 1 사이 값으로 정규화
        }
    
    def reset_if_idle(self): # 이 함수는 process_frame에서 손이 안보일 때 호출됨
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer.clear()
        # self.last_gesture_time.clear() # 필요시 쿨다운도 리셋


class NinjaGestureRecognizer:
    """닌자 게임 제스처 인식기 - 3제스처 버전"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        # OSC 설정
        self.client = udp_client.SimpleUDPClient(
            osc_ip or Config.OSC_IP, 
            osc_port or Config.OSC_PORT
        )

        # 안정화 모듈 설정
        if stabilizer_settings is None:
            stabilizer_settings = {}
        
        # 각 손에 대한 독립적인 안정화 모듈 생성
        self.stabilizers = {
            "Left": SimpleStabilizer(**stabilizer_settings),
            "Right": SimpleStabilizer(**stabilizer_settings)
        }
        # logger.info(f"Ninja Gesture Recognizer (3 Gestures + Position Tracking) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")
        # 위 로거는 NinjaMasterHandTracker로 이동하거나 중복 제거

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
        self.prev_time = {"Left": time.time(), "Right": time.time()} # 손별 시간 추적
        self.smoothed_landmarks = {"Left": None, "Right": None}
        
        # 위치 추적용 변수
        self.hand_positions = {"Left": "center", "Right": "center"}
        self.smoothed_hand_x = {"Left": 0.5, "Right": 0.5}

        # 웹캠 각도 보정을 위한 변수
        self.angle_correction_matrix = self._create_angle_correction_matrix()

        # 이 로거는 NinjaMasterHandTracker에서 호출하는 것으로 변경하거나, 내용을 통합하여 중복 제거
        logger.info(f"NinjaGestureRecognizer (3 Gestures) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")


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
        
        # self.hand_positions[hand_label] = position # 이 값은 recognize_gesture에서 사용됨
        return position, smoothed_x # smoothed_x도 반환하여 필요시 사용


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
        # Z축 회전 적용 (angle_correction_matrix 사용은 3D 좌표에 대해 이루어져야 함)
        # 현재는 Y 오프셋만 적용 중이므로, 매트릭스 사용은 주석 처리 또는 단순화된 형태로 남김
        # point = np.array([landmark.x, landmark.y, landmark.z])
        # corrected_point = self.angle_correction_matrix @ point 
        # corrected_x, corrected_y, corrected_z = corrected_point[0], corrected_point[1], corrected_point[2]
        
        # 단순 Y 오프셋 보정
        corrected_y = landmark.y - Config.Y_OFFSET_CORRECTION 
        return landmark.x, corrected_y, landmark.z # Z는 원래 값 유지 또는 보정된 Z 사용

    def _smooth_landmarks(self, current_landmarks, hand_label):
        """랜드마크 스무딩 적용"""
        if self.smoothed_landmarks[hand_label] is None or len(self.smoothed_landmarks[hand_label]) != len(current_landmarks):
            self.smoothed_landmarks[hand_label] = list(current_landmarks) # 리스트로 복사하여 사용
            return current_landmarks
        
        alpha = 0.7 # 스무딩 강도 (Config로 옮길 수 있음)
        smoothed_lm_list = []
        
        class SmoothLandmark: # 내부 클래스로 정의 또는 간단한 딕셔너리/namedtuple 사용
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        for i, curr_lm in enumerate(current_landmarks):
            prev_lm = self.smoothed_landmarks[hand_label][i]
            
            s_x = alpha * curr_lm.x + (1 - alpha) * prev_lm.x
            s_y = alpha * curr_lm.y + (1 - alpha) * prev_lm.y
            s_z = alpha * curr_lm.z + (1 - alpha) * prev_lm.z
            smoothed_lm_list.append(SmoothLandmark(s_x, s_y, s_z))
            
        self.smoothed_landmarks[hand_label] = smoothed_lm_list
        return smoothed_lm_list

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
        """손가락 굴곡 각도 계산 (2D 평면 기준)"""
        angles = {}
        # MCP, PIP, DIP 관절을 사용하여 각도 계산 (더 정확한 굽힘 감지)
        # 예: INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP
        # 현재 코드는 MCP, PIP, 중간 관절을 사용하는 것으로 보임
        finger_joints_indices = {
            'thumb': [self.mp_hands.HandLandmark.THUMB_CMC, self.mp_hands.HandLandmark.THUMB_MCP, self.mp_hands.HandLandmark.THUMB_IP, self.mp_hands.HandLandmark.THUMB_TIP],
            'index': [self.mp_hands.HandLandmark.INDEX_FINGER_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP, self.mp_hands.HandLandmark.INDEX_FINGER_DIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'middle': [self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring': [self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_PIP, self.mp_hands.HandLandmark.RING_FINGER_DIP, self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'pinky': [self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_PIP, self.mp_hands.HandLandmark.PINKY_DIP, self.mp_hands.HandLandmark.PINKY_TIP]
        }
        
        for finger, joint_indices in finger_joints_indices.items():
            # PIP 관절에서의 굽힘 각도 (MCP-PIP-DIP)
            p1_idx, p2_idx, p3_idx = joint_indices[0], joint_indices[1], joint_indices[2] 
            
            # 2D 좌표만 사용 (x, y)
            # landmark.x, landmark.y 사용
            # 엄지손가락은 다른 기준점을 사용할 수 있음 (예: WRIST-THUMB_CMC-THUMB_MCP)
            if finger == 'thumb': # 엄지는 MCP-IP-TIP 또는 WRIST-THUMB_MCP-THUMB_IP
                p1_lm = landmarks[self.mp_hands.HandLandmark.WRIST]
                p2_lm = landmarks[joint_indices[1]] # THUMB_MCP
                p3_lm = landmarks[joint_indices[2]] # THUMB_IP
            else:
                p1_lm = landmarks[p1_idx]
                p2_lm = landmarks[p2_idx]
                p3_lm = landmarks[p3_idx]

            # 2D 좌표만 사용
            p1 = np.array([p1_lm.x, p1_lm.y])
            p2 = np.array([p2_lm.x, p2_lm.y])
            p3 = np.array([p3_lm.x, p3_lm.y])
            
            angles[finger] = self.calculate_angle(p1, p2, p3)
        return angles

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지 - 공격 막기"""
        angles = self.calculate_finger_angles(landmarks)
        
        bent_fingers = 0
        total_relevant_fingers = 0 # 각도 계산이 가능한 손가락 수
        
        # 엄지를 제외한 4개 손가락 체크
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                total_relevant_fingers += 1
                # 주먹을 쥘 때 손가락 관절 각도는 작아짐
                if angles[finger] < Config.FIST_ANGLE_THRESHOLD: 
                    bent_fingers += 1
        
        # 3개 이상 손가락이 구부러지면 주먹 (엄지 제외)
        # 또는 모든 손가락(엄지 포함) 고려 시 bent_fingers >= 4
        if total_relevant_fingers > 0 and bent_fingers >= 3: # 4개 손가락 중 3개 이상
            confidence = 0.7 + (bent_fingers / total_relevant_fingers) * 0.3
            return True, confidence
        
        return False, 0.0

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지 - 표창 던지기 (위치 정보 포함)"""
        if self.prev_landmarks[hand_label] is None or len(self.prev_landmarks[hand_label]) != len(current_landmarks):
             self.prev_landmarks[hand_label] = list(current_landmarks) # 이전 랜드마크 업데이트
             return False, None, 0.0, "center" # 초기 상태이므로 flick 아님
        
        current_time = time.time()
        dt = current_time - self.prev_time[hand_label] # 손별 시간 사용

        if dt == 0 or dt > 0.5: # 너무 길거나 짧은 간격은 무시
            # self.prev_time[hand_label] = current_time # 시간 업데이트는 recognize_gesture 끝에서
            return False, None, 0.0, "center" 
        
        position, _ = self.calculate_hand_position(current_landmarks, hand_label)
        
        finger_tips_indices = [self.INDEX_TIP, self.MIDDLE_TIP]
        best_flick_direction = None
        max_velocity = 0.0
        
        for tip_idx in finger_tips_indices:
            # 현재 랜드마크와 이전 랜드마크에서 해당 손가락 끝점 가져오기
            curr_tip_lm = current_landmarks[tip_idx]
            prev_tip_lm = self.prev_landmarks[hand_label][tip_idx]

            # 좌표 보정 (카메라 각도 등) - 여기서는 Y 오프셋만 적용
            # 보정된 좌표는 이미지 좌표계로 변환하여 속도 계산에 사용
            # 여기서는 정규화된 좌표를 바로 사용하고, 속도 임계값을 조정하는 방식도 가능
            # 혹은 보정 없이 바로 사용하고 임계값으로 조절

            curr_x_norm, curr_y_norm, _ = self._correct_landmark_position(curr_tip_lm)
            prev_x_norm, prev_y_norm, _ = self._correct_landmark_position(prev_tip_lm)

            # 이미지 좌표로 변환 (속도 계산의 정확성을 위해)
            curr_pos_px = np.array([curr_x_norm * img_width, curr_y_norm * img_height])
            prev_pos_px = np.array([prev_x_norm * img_width, prev_y_norm * img_height])

            distance_px = self.calculate_distance(curr_pos_px, prev_pos_px)

            if distance_px < Config.MOVEMENT_THRESHOLD: # 너무 작은 움직임은 무시
                continue

            velocity = distance_px / dt # 픽셀/초 단위 속도

            if velocity > Config.FLICK_SPEED_THRESHOLD and velocity > max_velocity:
                direction_vector = curr_pos_px - prev_pos_px
                norm_direction = np.linalg.norm(direction_vector)
                
                if norm_direction > 0:
                    direction_normalized = direction_vector / norm_direction
                    
                    angle_rad = np.arctan2(direction_normalized[1], direction_normalized[0])
                    angle_deg = np.degrees(angle_rad)
                    
                    simplified_dir = [0.0, 0.0]
                    if -45 <= angle_deg < 45: simplified_dir = [1.0, 0.0]  # Right
                    elif 45 <= angle_deg < 135: simplified_dir = [0.0, 1.0] # Down
                    elif angle_deg >= 135 or angle_deg < -135: simplified_dir = [-1.0, 0.0] # Left
                    elif -135 <= angle_deg < -45: simplified_dir = [0.0, -1.0] # Up

                    # 손가락 펴짐 확인 (선택 사항, flick의 정의에 따라)
                    # finger_angles = self.calculate_finger_angles(current_landmarks)
                    # finger_name = 'index' if tip_idx == self.INDEX_TIP else 'middle'
                    # if finger_name in finger_angles and finger_angles[finger_name] > 120: # 펴짐 기준값
                    best_flick_direction = simplified_dir
                    max_velocity = velocity

        if best_flick_direction:
            confidence = min(0.7 + (max_velocity - Config.FLICK_SPEED_THRESHOLD) / 500, 1.0) # 속도에 따른 신뢰도
            return True, best_flick_direction, max_velocity, position
        
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
        # 입력된 hand_landmarks_obj.landmark는 MediaPipe의 NormalizedLandmarkList 객체이므로 리스트로 변환
        landmarks_list = list(hand_landmarks_obj.landmark)
        smoothed_landmarks_list = self._smooth_landmarks(landmarks_list, hand_label)
        
        height, width = img_shape[:2]
        
        current_gesture_type = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # Flick (가장 우선 순위)
        is_flick, flick_dir, flick_speed, flick_pos = self.detect_flick(
            smoothed_landmarks_list, hand_label, width, height
        )
        if is_flick:
            current_gesture_type = GestureType.FLICK
            gesture_data = {
                "direction": flick_dir, 
                "speed": flick_speed, 
                "confidence": min(0.7 + (flick_speed - Config.FLICK_SPEED_THRESHOLD) / 500.0, 1.0), # 재계산 또는 detect_flick에서 받은 값 사용
                "action": "throw_shuriken",
                "position": flick_pos
            }
        else:
            # Fist (두 번째 우선 순위)
            is_fist, fist_conf = self.detect_fist(smoothed_landmarks_list)
            if is_fist:
                current_gesture_type = GestureType.FIST
                gesture_data = {
                    "confidence": fist_conf,
                    "action": "block_attack",
                    # Fist는 위치 정보가 중요하지 않다면 생략하거나 "center"로 고정
                    "position": self.calculate_hand_position(smoothed_landmarks_list, hand_label)[0] 
                }
            else:
                # Palm Push (세 번째 우선 순위)
                is_push, push_conf, push_pos = self.detect_palm_push(smoothed_landmarks_list, hand_label)
                if is_push:
                    current_gesture_type = GestureType.PALM_PUSH
                    gesture_data = {
                        "confidence": push_conf,
                        "action": "shock_wave",
                        "position": push_pos
                    }

        # 현재 랜드마크를 다음 프레임의 이전 랜드마크로 저장 (리스트로 복사)
        self.prev_landmarks[hand_label] = list(landmarks_list) # 원본 랜드마크 저장
        self.prev_time[hand_label] = time.time() # 각 손의 마지막 처리 시간 업데이트

        return current_gesture_type, gesture_data


    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송 (위치 정보 포함)"""
        try:
            confidence = gesture_data.get("confidence", 0.0)
            gesture_type_str = gesture_type_enum.value
            action = gesture_data.get("action", "")
            # position은 fist의 경우 없을 수 있으므로 기본값 처리
            position = gesture_data.get("position", "center") 

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            # 기본 메시지 구성
            messages_to_send = {
                "/ninja/gesture/type": gesture_type_str,
                "/ninja/gesture/action": action,
                "/ninja/gesture/confidence": float(confidence),
                "/ninja/gesture/hand": hand_label,
            }

            # 위치 정보가 있는 경우 추가 (flick, palm_push)
            if gesture_type_str in ["flick", "palm_push"] and "position" in gesture_data:
                messages_to_send["/ninja/gesture/position"] = position
                messages_to_send["/ninja/gesture/position_action"] = f"{action}_{position}"

            # 방향 정보 (flick)
            if gesture_type_str == "flick" and "direction" in gesture_data:
                direction_val = gesture_data["direction"]
                if isinstance(direction_val, list) and len(direction_val) == 2:
                    # 방향 메시지는 별도로 추가 (하나의 메시지에 여러 arg 가능)
                    dir_msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                    dir_msg_builder.add_arg(float(direction_val[0]))
                    dir_msg_builder.add_arg(float(direction_val[1]))
                    bundle_builder.add_content(dir_msg_builder.build())

            # 속도 정보 (flick)
            if gesture_type_str == "flick" and "speed" in gesture_data:
                messages_to_send["/ninja/gesture/speed"] = float(gesture_data["speed"])

            # 메시지 빌드 및 추가
            for address, value in messages_to_send.items():
                if address == "/ninja/gesture/direction": continue # 이미 위에서 처리
                msg_builder = osc_message_builder.OscMessageBuilder(address=address)
                msg_builder.add_arg(value)
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
        """프레임 처리"""
        frame_to_draw_on = frame_input.copy()
        debug_messages_for_frame = []
        active_hands = 0 # 현재 프레임에서 감지된 손의 수

        try:
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                active_hands = len(results.multi_hand_landmarks)
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    handedness_obj = results.multi_handedness[hand_idx]
                    hand_label = handedness_obj.classification[0].label # "Left" or "Right"
                    
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    raw_gesture_type, raw_gesture_data = self.recognize_gesture(
                        hand_landmarks_obj, hand_label, frame_input.shape
                    )
                    
                    # 각 손에 맞는 안정화기 사용
                    stabilizer_for_hand = self.stabilizers[hand_label]
                    should_send, _ = stabilizer_for_hand.should_send_gesture( # stabilized_gesture_data는 현재 미사용
                        raw_gesture_type.value,
                        raw_gesture_data.get("confidence", 0.0),
                        hand_label
                    )
                    
                    if should_send and raw_gesture_type != GestureType.NONE:
                        # should_send가 True이면, raw_gesture_data를 사용 (안정화기 내부에서 최종 제스처 결정)
                        # 이때 raw_gesture_type은 stabilizer_for_hand.current_gesture를 Enum으로 변환하여 사용 가능
                        final_gesture_type_enum = GestureType(stabilizer_for_hand.current_gesture)
                        # gesture_data는 raw_gesture_data를 그대로 사용하거나, 안정화된 제스처에 맞게 필터링
                        # 여기서는 raw_gesture_data를 사용하되, 안정화된 타입과 일치하는지 확인 필요
                        # 만약 raw_gesture_type.value != stabilizer_for_hand.current_gesture 이면,
                        # 안정화기가 다른 제스처를 최종 판단한 것이므로, 해당 제스처의 데이터를 찾아야 함.
                        # 지금은 raw_gesture_data를 그대로 사용하고, OSC 전송 시 타입은 안정화된 것을 사용.
                        self.send_gesture_osc(final_gesture_type_enum, raw_gesture_data, hand_label)
                        action = raw_gesture_data.get("action", "")
                        pos_info = f" @ {raw_gesture_data.get('position','N/A').upper()}" if final_gesture_type_enum in [GestureType.FLICK, GestureType.PALM_PUSH] else ""
                        debug_messages_for_frame.append(f"{hand_label}: {final_gesture_type_enum.value} ({action}){pos_info} ✓")
                    else:
                        stabilizer_stats = stabilizer_for_hand.get_statistics()
                        pending_gesture = stabilizer_stats.get("current_gesture", "none")
                        
                        if pending_gesture != "none":
                            progress_percent = stabilizer_stats.get("stability_progress", 0) * 100
                            # raw_gesture_data에서 현재 pending_gesture와 관련된 action, position 가져오기 (정확도 향상 위해)
                            current_action_display = raw_gesture_data.get("action", "") if raw_gesture_type.value == pending_gesture else ""
                            current_pos_display = f" @ {raw_gesture_data.get('position','N/A').upper()}" if raw_gesture_type.value == pending_gesture and pending_gesture in ["flick", "palm_push"] else ""
                            
                            debug_messages_for_frame.append(
                                f"{hand_label}: {pending_gesture} ({current_action_display}{current_pos_display}) ({progress_percent:.0f}%)"
                            )
            
            # 손이 감지되지 않은 경우, 각 안정기 리셋
            if active_hands == 0:
                for stabilizer in self.stabilizers.values():
                    stabilizer.reset_if_idle()
            elif active_hands == 1: # 한 손만 감지된 경우, 다른 손의 안정기 리셋
                detected_hand_label = results.multi_handedness[0].classification[0].label
                for label, stabilizer in self.stabilizers.items():
                    if label != detected_hand_label:
                        stabilizer.reset_if_idle()


            self.send_hand_state(active_hands)

        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}", exc_info=True) # exc_info=True로 트레이스백 로깅
            debug_messages_for_frame.append(f"Error: {str(e)[:50]}") # 오류 메시지 간략화

        # self.prev_time은 recognize_gesture 내부에서 손별로 업데이트되므로 여기서 전체 prev_time 업데이트는 제거
        return frame_to_draw_on, debug_messages_for_frame

    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        logger.info("Gesture recognizer cleaned up.")


class NinjaMasterHandTracker:
    """메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        # Gesture Recognizer 인스턴스 생성
        # NinjaGestureRecognizer가 OSC 클라이언트 및 Stabilizer를 내부적으로 관리
        self.gesture_recognizer = NinjaGestureRecognizer(
            osc_ip=osc_ip,
            osc_port=osc_port,
            stabilizer_settings=stabilizer_settings
        )
        logger.info(f"NinjaMasterHandTracker initialized using NinjaGestureRecognizer.")
        logger.info(f"OSC settings for NGR: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")
        logger.info(f"Stabilizer settings for NGR: {stabilizer_settings or 'Defaults'}")


        # 카메라 설정
        self.cap = cv2.VideoCapture(0) # 0번 카메라 사용, 필요시 변경
        if not self.cap.isOpened():
            logger.critical("카메라를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
            raise IOError("카메라를 열 수 없습니다.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE) # 버퍼 크기 설정
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera requested: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT} @{Config.CAMERA_FPS}FPS")
        logger.info(f"Camera actual: {actual_width}x{actual_height} @{actual_fps}FPS")


        # FPS 계산용 변수
        self.last_time = time.time()
        self.fps_counter = deque(maxlen=Config.FPS_BUFFER_SIZE) # FPS 표시용 평균 계산

        # 디버그 모드
        self.debug_mode = True # 기본적으로 디버그 모드 활성화

        # 아래 변수들은 NinjaGestureRecognizer로 이전되었으므로 NinjaMasterHandTracker에서 제거
        # self.client = udp_client.SimpleUDPClient(...)
        # self.stabilizer = SimpleStabilizer(...)
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(...)
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.WRIST, self.THUMB_TIP, ...
        # self.prev_landmarks, self.smoothed_landmarks
        # self.hand_positions, self.smoothed_hand_x
        # self.angle_correction_matrix

        # 중복 로거 호출 제거
        # logger.info(f"Ninja Gesture Recognizer (3 Gestures + Position Tracking) initialized - OSC: ...")

    
    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        time_difference = current_time - self.last_time
        
        if time_difference > 0: # 0으로 나누기 방지
            fps = 1.0 / time_difference
            self.fps_counter.append(fps)
        
        self.last_time = current_time # 다음 계산을 위해 현재 시간 저장
        
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
        for y_coord in range(0, height, 20): # 변수명 변경
            cv2.line(frame, (left_line, y_coord), (left_line, y_coord + 10), (100, 100, 100), 2)
            cv2.line(frame, (right_line, y_coord), (right_line, y_coord + 10), (100, 100, 100), 2)
        
        # 영역 라벨
        cv2.putText(frame, "LEFT", (left_line // 4, 50), # 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "CENTER", (left_line + (right_line - left_line) // 2 - 40, 50), # 중앙 정렬
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (right_line + (width - right_line) // 4, 50), # 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        
        # 제스처 정보
        y_offset = 90
        for message in debug_messages_list:
            color = (0, 255, 255) if "✓" in message else (255, 255, 0) # 확인된 제스처는 다른 색
            cv2.putText(frame, message, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y_offset += 30
        
        # 제스처 가이드 - 위치 정보 추가
        guide_y = height - 220 # 위치 조정
        cv2.putText(frame, "=== 3 Core Gestures + Position ===", (10, guide_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        gestures = [
            ("1. FLICK", "Quick finger movement", "Throw Shuriken (L/C/R)", (255, 100, 100)),
            ("2. FIST", "Close your hand", "Block Attack (Pos: Auto)", (100, 255, 100)), # Fist 위치는 자동 계산됨을 명시
            ("3. PALM PUSH", "Open all 5 fingers", "Shock Wave (L/C/R)", (100, 100, 255))
        ]
        
        for i, (gesture, desc, action, color) in enumerate(gestures):
            y_pos = guide_y + 30 + (i * 25)
            cv2.putText(frame, f"{gesture}: {desc} => {action}", (10, y_pos), # => 로 변경
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # 위치 추적 정보
        cv2.putText(frame, "Position Tracking: LEFT | CENTER | RIGHT", (10, height - 100), # 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
            
        # 하단 정보
        cv2.putText(frame, "Ninja Master - 3 Gesture + Position System", (10, height - 70), # 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, height - 40), # 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)


    def run(self):
        """메인 루프"""
        logger.info("Starting Ninja Master - Simple 3 Gesture System...")
        logger.info("Gestures: FLICK (표창), FIST (방어), PALM (진동파)")
        
        try:
            while True:
                success, frame_from_camera = self.cap.read()
                if not success or frame_from_camera is None: # 프레임 None 체크 추가
                    logger.error("웹캠에서 프레임을 읽을 수 없습니다. 루프를 종료합니다.")
                    time.sleep(0.5) # 잠시 대기 후 재시도 또는 종료
                    break # 또는 continue로 다음 프레임 시도

                current_frame_flipped = cv2.flip(frame_from_camera, 1) # 좌우 반전
                
                # gesture_recognizer의 process_frame 호출
                processed_display_frame, current_debug_messages = self.gesture_recognizer.process_frame(current_frame_flipped)
                
                current_fps = self.calculate_fps()
                
                if self.debug_mode:
                    self.draw_debug_info(processed_display_frame, current_fps, current_debug_messages)
                
                cv2.imshow("Ninja Master", processed_display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("종료합니다 (q 입력).")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}", exc_info=True) # 트레이스백 포함
            import traceback
            traceback.print_exc() # 콘솔에도 트레이스백 출력
            
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        logger.info("리소스 정리 중...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            logger.info("카메라 리소스 해제됨.")
        cv2.destroyAllWindows()
        logger.info("모든 OpenCV 창 닫힘.")
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup() # NinjaGestureRecognizer의 cleanup 호출
            logger.info("Gesture Recognizer 리소스 정리됨.")
        logger.info("프로그램 종료.")


def test_mode():
    """OSC 통신 테스트 모드 (위치 정보 포함)"""
    try:
        # test_osc_communication.py가 같은 디렉토리에 있다고 가정
        from test_osc_communication import OSCTester 
        
        logger.info("=== OSC 테스트 모드 (3 Gestures + Position) ===")
        tester = OSCTester(ip=Config.OSC_IP, port=Config.OSC_PORT) # IP, Port 전달
        tester.start_server() # 서버 시작

        # 테스트용 클라이언트 (NinjaGestureRecognizer와 동일한 설정 사용)
        test_client = udp_client.SimpleUDPClient(Config.OSC_IP, Config.OSC_PORT)
        logger.info(f"Test OSC Client_결과 전송 대상: {Config.OSC_IP}:{Config.OSC_PORT}")


        while True:
            print("\n테스트 옵션:")
            print("1. 3가지 제스처 테스트 (Flick, Fist, Palm Push - Center)")
            print("2. 방향별 플릭 테스트 (Up, Down, Left, Right - Center)")
            print("3. 위치별 제스처 테스트 (Flick, Palm Push - Left, Center, Right)")
            print("4. 개별 제스처 상세 테스트 (수동 입력)")
            print("5. 종료")
            
            choice = input("선택: ")
            
            hand = "Right" # 기본 손
            confidence = 0.95
            speed = 350.0

            if choice == "1":
                gestures_to_test = {
                    "flick": {"action": "throw_shuriken", "position": "center", "direction": [1.0, 0.0], "speed": speed},
                    "fist": {"action": "block_attack", "position": "center"}, # Fist는 방향/속도 없음
                    "palm_push": {"action": "shock_wave", "position": "center"} # Palm Push는 방향/속도 없음
                }
                for gesture_type, data in gestures_to_test.items():
                    print(f"\n테스트: {gesture_type.upper()} ({data['action']}) @ {data['position']}")
                    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
                    
                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type")
                    msg.add_arg(gesture_type)
                    bundle.add_content(msg.build())

                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/action")
                    msg.add_arg(data["action"])
                    bundle.add_content(msg.build())
                    
                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position")
                    msg.add_arg(data["position"])
                    bundle.add_content(msg.build())

                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position_action")
                    msg.add_arg(f"{data['action']}_{data['position']}")
                    bundle.add_content(msg.build())

                    if "direction" in data:
                        msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/direction")
                        msg.add_arg(float(data["direction"][0]))
                        msg.add_arg(float(data["direction"][1]))
                        bundle.add_content(msg.build())
                    if "speed" in data:
                        msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/speed")
                        msg.add_arg(float(data["speed"]))
                        bundle.add_content(msg.build())

                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence")
                    msg.add_arg(confidence)
                    bundle.add_content(msg.build())

                    msg = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand")
                    msg.add_arg(hand)
                    bundle.add_content(msg.build())
                    
                    test_client.send(bundle.build())
                    time.sleep(1)


            elif choice == "2":
                print("\n- 방향별 플릭 테스트 (모두 CENTER 위치에서)")
                directions = {"right": [1.0, 0.0], "left": [-1.0, 0.0], "up": [0.0, -1.0], "down": [0.0, 1.0]}
                for dir_name, dir_vec in directions.items():
                    print(f"  CENTER 플릭 (방향: {dir_name.upper()})")
                    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
                    # ... (flick 메시지 구성, choice "1"과 유사하게, direction만 변경)
                    test_client.send(bundle.build()) # 여기에 전체 번들 전송 로직 필요
                    time.sleep(1)
                pass # 상세 구현 필요

            elif choice == "3":
                print("\n위치별 제스처 테스트:")
                positions = ["left", "center", "right"]
                
                # Flick 위치별 테스트
                print("\n- FLICK 위치별 테스트 (방향: 오른쪽)")
                for position in positions:
                    print(f"  {position.upper()} 플릭 (오른쪽)")
                    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
                    # ... (flick 메시지 구성, choice "1"과 유사하게, position만 변경)
                    test_client.send(bundle.build()) # 여기에 전체 번들 전송 로직 필요
                    time.sleep(1)

                # Palm Push 위치별 테스트
                print("\n- PALM PUSH 위치별 테스트")
                for position in positions:
                    print(f"  {position.upper()} 진동파")
                    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
                    # ... (palm_push 메시지 구성, choice "1"과 유사하게, position만 변경)
                    test_client.send(bundle.build()) # 여기에 전체 번들 전송 로직 필요
                    time.sleep(1)
                pass # 상세 구현 필요
                    
            elif choice == "4":
                # 개별 제스처 상세 테스트 (사용자 입력 받아 전송)
                print("\n- 개별 제스처 상세 테스트")
                g_type = input("제스처 타입 (flick, fist, palm_push): ").lower()
                g_action = input("액션 (throw_shuriken, block_attack, shock_wave): ").lower()
                g_pos = input("위치 (left, center, right): ").lower()
                g_hand = input("손 (Left, Right): ").capitalize()
                # ... 추가 정보 입력 (방향, 속도 등)
                # ... OSC 메시지 구성 및 전송 ...
                pass # 상세 구현 필요
            
            elif choice == "5":
                break
        
        tester.stop_server()
        logger.info("테스트 모드 종료.")

    except ImportError:
        logger.error("test_osc_communication.py를 찾을 수 없습니다. 테스트 모드를 실행하려면 해당 파일이 필요합니다.")
    except Exception as e:
        logger.error(f"테스트 모드 오류: {e}", exc_info=True)


if __name__ == "__main__":
    import sys
    
    is_test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    if is_test_mode:
        test_mode()
    else:
        # 3 제스처 시스템에 최적화된 설정
        custom_stabilizer_settings = {
            "stability_window": Config.DEFAULT_STABILITY_WINDOW, # Config 값 사용
            "confidence_threshold": Config.DEFAULT_CONFIDENCE_THRESHOLD,
            "cooldown_time": Config.DEFAULT_COOLDOWN_TIME 
        }
        
        print("\n")
        print("=" * 50)
        print("    닌자 마스터 - 3 제스처 시스템 (+ 위치)")
        print("=" * 50)
        print("\n핵심 제스처:")
        print("  1. FLICK     - 손가락 튕기기 → 표창 던지기 (L/C/R)")
        print("  2. FIST      - 주먹 쥐기     → 공격 막기 (위치 자동)")
        print("  3. PALM PUSH - 5손가락 펴기  → 진동파 (L/C/R)")
        print("\n조작법:")
        print("  • Q - 종료")
        print("  • D - 디버그 모드 전환")
        print("=" * 50)
        print("\n")
        
        tracker_instance = None # finally에서 사용하기 위해 미리 선언
        try:
            # NinjaMasterHandTracker 생성 시 osc_ip, osc_port도 명시적으로 전달하거나
            # Config 기본값을 사용하도록 생성자에서 처리
            tracker_instance = NinjaMasterHandTracker(
                osc_ip=Config.OSC_IP,
                osc_port=Config.OSC_PORT,
                stabilizer_settings=custom_stabilizer_settings # 수정된 파라미터명
            )
            tracker_instance.run()
        except IOError as e: # 카메라 관련 오류 등
            logger.critical(f"시작 실패 (IOError): {e}. 카메라 연결 또는 권한을 확인하세요.")
        except KeyboardInterrupt:
            logger.info("\n사용자 중단 (Ctrl+C).")
        except Exception as e:
            logger.critical(f"치명적 오류 발생: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
        finally:
            if tracker_instance: # tracker_instance가 성공적으로 생성된 경우에만 cleanup 호출
                 logger.info("프로그램 종료 절차 시작...")
                 tracker_instance.cleanup()
            else:
                 logger.info("트래커 인스턴스가 생성되지 않아 추가 정리 작업 없음.")
            logger.info("프로그램이 완전히 종료되었습니다.")