# gesture_recognizer.py - 닌자 게임 인식 시스템 (듀얼 제스처 개선 버전)

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
    """제스처 타입 정의 - 2가지 제스처 지원"""
    NONE = "none"
    FLICK = "flick"          # 표창 던지기 (아래에서 위로)
    FIST = "fist"            # 주먹 쥐기


class Config:
    """설정값 관리 - 듀얼 제스처를 위해 조정됨"""
    # OSC 설정
    OSC_IP = "127.0.0.1"
    OSC_PORT = 7000

    # MediaPipe 설정 - 정확도 우선
    MAX_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5  # 0.6에서 하향 조정
    MIN_TRACKING_CONFIDENCE = 0.4   # 0.5에서 하향 조정
    MODEL_COMPLEXITY = 1

    # 카메라 설정
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1

    # 웹캠 위치 보정 (머리 위)
    CAMERA_ANGLE_CORRECTION = 15
    Y_OFFSET_CORRECTION = 0.1

    # 제스처 임계값 - FLICK 인식 개선을 위해 조정
    FLICK_SPEED_THRESHOLD = 120      # 150에서 120으로 더 하향 (더 민감하게)
    FLICK_VERTICAL_RATIO = 0.4        # 0.5에서 0.4로 하향 (수직 움직임 비율 더 완화)
    FIST_ANGLE_THRESHOLD = 110        # 90에서 110으로 상향 (더 관대하게)
    FIST_FINGER_DISTANCE_THRESHOLD = 0.06  # 주먹 손가락 끝점 거리 임계값 추가

    # Flick 정확도 - 검지와 중지 거리 (더 관대하게)
    FLICK_FINGER_DISTANCE_THRESHOLD = 0.06  # 0.05에서 0.06으로 상향

    # 안정화 설정 - FLICK을 위해 더 빠르게
    DEFAULT_STABILITY_WINDOW = 0.15     # 0.1에서 0.15로 약간 상향 (FIST 인식 개선)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.55 # 0.6에서 0.55로 하향 (더 관대하게)
    DEFAULT_COOLDOWN_TIME = 0.25        # 0.2에서 0.25로 약간 상향 (안정성)
    
    # FIST 전용 안정화 설정
    FIST_STABILITY_WINDOW = 0.4         # FIST는 더 긴 유지시간 필요
    FIST_COOLDOWN_TIME = 0.5            # FIST는 더 긴 쿨다운

    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 5  # 7에서 5로 감소 (더 빠른 반응)
    FPS_BUFFER_SIZE = 30

    # 노이즈 필터링
    MOVEMENT_THRESHOLD = 8  # 10에서 8로 하향 (더 작은 움직임도 감지)
    GESTURE_CHANGE_THRESHOLD = 0.4  # 0.5에서 0.4로 하향
    POSITION_CHANGE_THRESHOLD = 0.2

    # 위치 트래킹 설정
    POSITION_LEFT_THRESHOLD = 0.33
    POSITION_RIGHT_THRESHOLD = 0.66
    POSITION_TRACKING_SMOOTHING = 0.7

    # FLICK 개선을 위한 추가 설정
    FLICK_ANGLE_TOLERANCE = 35  # 30에서 35로 증가 (더 관대하게)
    FLICK_MIN_DISTANCE = 25     # 30에서 25로 감소 (더 작은 움직임도 인식)


class EnhancedStabilizer:
    """개선된 듀얼 제스처 안정화기"""
    def __init__(self, **kwargs):
        self.stability_window = kwargs.get('stability_window', Config.DEFAULT_STABILITY_WINDOW)
        self.confidence_threshold = kwargs.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        self.cooldown_time = kwargs.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
        self.last_gesture_time = {}
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer = deque(maxlen=4)  # 6에서 4로 더 감소 (더 빠른 반응)
        self.position_buffer = deque(maxlen=5)
        self.last_valid_position = "center"
        self.last_sent_gesture = "none"
        self.confidence_buffer = deque(maxlen=3)  # 5에서 3으로 감소

    def should_send_gesture(self, gesture_type, confidence, hand_label, position=None):
        current_time = time.time()

        if gesture_type == "none":
            return False, None

        # 제스처별 다른 설정 적용
        if gesture_type == "fist":
            stability_window = Config.FIST_STABILITY_WINDOW
            cooldown_time = Config.FIST_COOLDOWN_TIME
        else:
            stability_window = self.stability_window
            cooldown_time = self.cooldown_time

        # 신뢰도 버퍼링 및 평균 계산
        self.confidence_buffer.append(confidence)
        avg_confidence = np.mean(self.confidence_buffer) if self.confidence_buffer else confidence

        # 평균 신뢰도 체크
        if avg_confidence < self.confidence_threshold:
            return False, None

        # 위치 안정화
        if position:
            self.position_buffer.append(position)
            if len(self.position_buffer) >= 3:
                position_counts = {}
                for pos in self.position_buffer:
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                most_common_position = max(position_counts, key=position_counts.get)
                
                if position_counts[most_common_position] >= len(self.position_buffer) * 0.5:  # 0.6에서 0.5로
                    self.last_valid_position = most_common_position
            position = self.last_valid_position

        self.gesture_buffer.append(gesture_type)

        # 버퍼의 50% 이상이 같은 제스처면 인식 (60%에서 하향)
        if len(self.gesture_buffer) >= 2:  # 4에서 2로 감소
            gesture_count = {}
            for g in self.gesture_buffer:
                gesture_count[g] = gesture_count.get(g, 0) + 1

            most_common = max(gesture_count, key=gesture_count.get)
            if gesture_count[most_common] >= len(self.gesture_buffer) * 0.5:  # 0.6에서 0.5로
                gesture_type = most_common
            else:
                return False, None
        else:
            return False, None

        # 새로운 제스처 시작
        if gesture_type != self.current_gesture:
            self.current_gesture = gesture_type
            self.current_gesture_start = current_time
            self.confidence_buffer.clear()
            return False, None

        # 안정화 시간 체크 (제스처별 다른 시간 적용)
        if current_time - self.current_gesture_start < stability_window:
            return False, None

        # 쿨다운 체크 (제스처별 다른 쿨다운 적용)
        gesture_key = f"{gesture_type}_{hand_label}"
        if gesture_key in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_key] < cooldown_time:
                return False, None

        # 제스처 전송
        self.last_gesture_time[gesture_key] = current_time
        self.gesture_buffer.clear()
        self.confidence_buffer.clear()
        self.last_sent_gesture = gesture_type

        return True, {"confidence": avg_confidence}

    def get_statistics(self):
        stability_progress = 0
        if self.current_gesture != "none":
            elapsed = time.time() - self.current_gesture_start
            # 제스처별 다른 stability window 적용
            if self.current_gesture == "fist":
                stability_window = Config.FIST_STABILITY_WINDOW
            else:
                stability_window = self.stability_window
            stability_progress = min(elapsed / stability_window, 1.0)
        
        return {
            "current_gesture": self.current_gesture,
            "stability_progress": stability_progress,
            "confidence_avg": np.mean(self.confidence_buffer) if self.confidence_buffer else 0
        }

    def reset_if_idle(self):
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer.clear()
        self.confidence_buffer.clear()


class NinjaGestureRecognizer:
    """닌자 게임 제스처 인식기 - 듀얼 제스처 개선 버전"""

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
        logger.info(f"Enhanced Dual-Gesture Stabilizer initialized with settings: {stabilizer_settings}")

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

        # 추가 랜드마크
        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

        # 상태 추적용 변수
        self.prev_landmarks = {"Left": None, "Right": None}
        self.prev_time = time.time()
        self.smoothed_landmarks = {"Left": None, "Right": None}

        # 위치 추적용 변수
        self.hand_positions = {"Left": "center", "Right": "center"}
        self.smoothed_hand_x = {"Left": 0.5, "Right": 0.5}

        # 웹캠 각도 보정을 위한 변수
        self.angle_correction_matrix = self._create_angle_correction_matrix()
        
        # 추가 버퍼 - FLICK 개선
        self.velocity_buffer = deque(maxlen=3)  # 5에서 3으로 감소 (더 빠른 반응)
        self.flick_history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}  # 10에서 5로 감소

        logger.info(f"Ninja Gesture Recognizer (Dual-Gesture) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

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

        alpha = 0.8  # 0.7에서 0.8로 증가 (현재 값에 더 가중치)
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
        middle_mcp_x = landmarks[9].x
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

    def check_finger_tips_distance(self, landmarks):
        """검지와 중지 끝점 사이의 거리 체크 (Flick용)"""
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        distance = self.calculate_distance(
            [index_tip.x, index_tip.y],
            [middle_tip.x, middle_tip.y]
        )

        return distance < Config.FLICK_FINGER_DISTANCE_THRESHOLD, distance

    def check_all_finger_tips_close(self, landmarks):
        """모든 손가락 끝점이 가까운지 체크 (Fist용)"""
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        
        # 손가락 끝점들 간의 평균 거리 계산
        distances = []
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                tip1 = landmarks[finger_tips[i]]
                tip2 = landmarks[finger_tips[j]]
                distance = self.calculate_distance(
                    [tip1.x, tip1.y],
                    [tip2.x, tip2.y]
                )
                distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 1.0
        max_distance = np.max(distances) if distances else 1.0
        
        # 모든 손가락이 가까운지 확인
        all_close = max_distance < Config.FIST_FINGER_DISTANCE_THRESHOLD
        
        return all_close, avg_distance, max_distance

    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지 - 개선된 버전"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0.0, "center"

        current_time = time.time()
        dt = current_time - self.prev_time
        if dt == 0 or dt > 0.5:
            return False, None, 0.0, "center"

        position, _ = self.calculate_hand_position(current_landmarks, hand_label)
        
        # 검지와 중지 끝이 가까운지 확인 (더 관대한 조건)
        fingers_together, finger_distance = self.check_finger_tips_distance(current_landmarks)
        
        # 손가락이 완전히 붙어있지 않아도 어느 정도 가까우면 OK
        if finger_distance > Config.FLICK_FINGER_DISTANCE_THRESHOLD * 1.5:
            self.velocity_buffer.clear()
            return False, None, 0.0, position

        # 여러 손가락의 움직임을 추적
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP]  # 약지도 포함
        best_flick = None
        best_velocity = 0
        best_vertical_ratio = 0

        flick_candidates = []

        for finger_tip in finger_tips:
            curr_tip = current_landmarks[finger_tip]
            prev_tip = self.prev_landmarks[hand_label][finger_tip]

            curr_x, curr_y, _ = self._correct_landmark_position(curr_tip)
            prev_x, prev_y, _ = self._correct_landmark_position(prev_tip)

            # Y축은 위로 갈수록 작아짐
            delta_x = (curr_x - prev_x) * img_width
            delta_y = (curr_y - prev_y) * img_height  # 음수면 위로 움직임
            
            distance = self.calculate_distance([0, 0], [delta_x, delta_y])

            # 최소 이동 거리 체크
            if distance < Config.FLICK_MIN_DISTANCE:
                continue
            
            velocity = distance / dt
            
            # 수직 움직임 비율 계산
            vertical_ratio = abs(delta_y) / (distance + 1e-6)
            upward_movement = delta_y < 0  # Y가 감소하면 위로 움직임
            
            # 움직임 각도 계산
            angle_from_vertical = abs(np.degrees(np.arctan2(abs(delta_x), abs(delta_y))))
            
            # 조건 완화: 각도 허용치 내에 있고 위로 움직이면 OK
            if (upward_movement and 
                angle_from_vertical < Config.FLICK_ANGLE_TOLERANCE and
                vertical_ratio > Config.FLICK_VERTICAL_RATIO):
                
                flick_candidates.append({
                    'finger': finger_tip,
                    'velocity': velocity,
                    'vertical_ratio': vertical_ratio,
                    'direction': [delta_x / distance, delta_y / distance],
                    'angle': angle_from_vertical
                })

        # 후보 중에서 최적의 FLICK 선택
        if flick_candidates:
            # 속도와 수직 비율을 고려한 점수 계산
            best_candidate = max(flick_candidates, 
                               key=lambda x: x['velocity'] * 0.7 + x['vertical_ratio'] * 1000 * 0.3)
            
            self.velocity_buffer.append(best_candidate['velocity'])
            
            # 평균 속도 계산
            if len(self.velocity_buffer) >= 2:
                avg_velocity = np.mean(list(self.velocity_buffer)[-2:])  # 최근 2개만 사용
            else:
                avg_velocity = best_candidate['velocity']

            # 히스토리에 추가
            self.flick_history[hand_label].append({
                'time': current_time,
                'velocity': avg_velocity,
                'vertical_ratio': best_candidate['vertical_ratio']
            })

            # 최근 패턴 분석
            recent_history = [h for h in self.flick_history[hand_label] 
                            if current_time - h['time'] < 0.3]  # 0.5에서 0.3으로 감소
            
            # 속도 임계값 동적 조정
            dynamic_threshold = Config.FLICK_SPEED_THRESHOLD
            if len(recent_history) > 1:  # 2에서 1로 감소
                # 최근에 빠른 움직임이 있었다면 임계값을 낮춤
                max_recent_velocity = max(h['velocity'] for h in recent_history)
                if max_recent_velocity > Config.FLICK_SPEED_THRESHOLD * 1.2:  # 1.5에서 1.2로
                    dynamic_threshold *= 0.7  # 0.8에서 0.7로

            # 최종 조건 확인
            if avg_velocity > dynamic_threshold:
                best_flick = best_candidate['direction']
                best_velocity = avg_velocity
                best_vertical_ratio = best_candidate['vertical_ratio']

                # 신뢰도 계산 개선
                velocity_score = min((avg_velocity - dynamic_threshold) / 400, 0.3)  # 500에서 400으로
                vertical_score = min(best_vertical_ratio * 0.3, 0.3)
                distance_score = min((1.0 - (finger_distance / Config.FLICK_FINGER_DISTANCE_THRESHOLD)) * 0.2, 0.2)
                angle_score = min((1.0 - best_candidate['angle'] / Config.FLICK_ANGLE_TOLERANCE) * 0.2, 0.2)
                
                confidence = 0.6 + velocity_score + vertical_score + distance_score + angle_score  # 0.5에서 0.6으로
                confidence = min(confidence, 1.0)
                
                logger.info(f"Flick detected! Velocity: {best_velocity:.1f}, Vertical: {best_vertical_ratio:.2f}, Angle: {best_candidate['angle']:.1f}°")
                return True, best_flick, best_velocity, position
        
        return False, None, 0.0, position

    def detect_fist(self, landmarks):
        """주먹 쥐기 감지 - 3개 이상 손가락 굽히면 인식, 스마트 충돌 방지"""
        angles = self.calculate_finger_angles(landmarks)
        bent_fingers = 0
        total_fingers = 0
        bent_finger_names = []
        
        # 엄지를 제외한 네 손가락 확인
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles:
                total_fingers += 1
                if angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                    bent_fingers += 1
                    bent_finger_names.append(finger)
        
        # 3개 이상 손가락이 굽혀져 있으면 주먹으로 인식
        if total_fingers >= 4 and bent_fingers >= 3:
            # Flick과의 충돌을 더 정교하게 체크
            # 검지와 중지가 모두 굽혀져 있고, 매우 가까운 경우만 체크
            if 'index' in bent_finger_names and 'middle' in bent_finger_names:
                fingers_together, finger_distance = self.check_finger_tips_distance(landmarks)
                # 거리가 매우 가까우면서 움직임이 있을 때만 Flick 가능성으로 판단
                if finger_distance < Config.FLICK_FINGER_DISTANCE_THRESHOLD * 0.7:
                    # 여기서는 패스하고 움직임은 recognize_gesture에서 체크
                    pass
            
            # 신뢰도 계산 (더 관대하게)
            base_confidence = 0.6  # 기본 신뢰도 상향
            angle_bonus = (bent_fingers / total_fingers) * 0.3
            
            # 4개 모두 굽혔으면 추가 보너스
            if bent_fingers == 4:
                angle_bonus += 0.1
                
            confidence = base_confidence + angle_bonus
            confidence = min(confidence, 1.0)
            
            logger.info(f"Fist detected! Bent: {bent_fingers}/4 ({', '.join(bent_finger_names)}) Conf: {confidence:.2f}")
            return True, confidence
        
        return False, 0.0

    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        """통합 제스처 인식 - 우선순위: FLICK > FIST"""
        landmarks = self._smooth_landmarks(hand_landmarks_obj.landmark, hand_label)
        height, width = img_shape[:2]
    
        current_gesture = GestureType.NONE
        gesture_data = {"confidence": 0.0}

        # 우선순위: FLICK > FIST
        # FLICK 제스처는 손가락을 모으는 동작이 포함되어 FIST와 충돌할 수 있음
        # 따라서 FLICK을 먼저 검사하고, FLICK이 아닐 때만 FIST 검사
        
        # 1. FLICK 검사 (양손 모두)
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
                "position": flick_position
            }
        else:
            # 2. FLICK이 아닐 때만 FIST 검사
            # FIST는 3개 이상 손가락이 굽혀져야 함
            is_fist, fist_conf = self.detect_fist(landmarks)
            if is_fist:
                current_gesture = GestureType.FIST
                gesture_data = {
                    "confidence": fist_conf,
                    "action": "activate_fist"
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

            # 위치 정보 (FLICK에만 해당)
            if gesture_type_str == "flick":
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position")
                msg_builder.add_arg(position)
                bundle_builder.add_content(msg_builder.build())
                
                position_action = f"{action}_{position}"
                msg_builder = osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position_action")
                msg_builder.add_arg(position_action)
                bundle_builder.add_content(msg_builder.build())

            # 방향 및 속도 정보 (FLICK용)
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
            if position and gesture_type_str == "flick":
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

                    # 제스처별 시각화
                    
                    # FLICK: 검지-중지 거리 시각화
                    index_tip = landmarks[self.INDEX_TIP]
                    middle_tip = landmarks[self.MIDDLE_TIP]
                    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                    middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

                    is_close_for_flick, finger_dist = self.check_finger_tips_distance(landmarks)
                    
                    # 거리에 따른 색상 변경
                    if finger_dist < Config.FLICK_FINGER_DISTANCE_THRESHOLD:
                        color = (0, 255, 0)  # 초록색
                        text = "FLICK RDY"
                    elif finger_dist < Config.FLICK_FINGER_DISTANCE_THRESHOLD * 1.5:
                        color = (0, 255, 255)  # 노란색
                        text = "ALMOST"
                    else:
                        color = (0, 0, 255)  # 빨간색
                        text = f"{finger_dist:.3f}"
                    
                    cv2.line(frame_to_draw_on, index_pos, middle_pos, color, 3)
                    cv2.putText(frame_to_draw_on, text, 
                               ((index_pos[0] + middle_pos[0]) // 2 - 30, 
                                (index_pos[1] + middle_pos[1]) // 2 - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # FIST: 손가락 굽힘 상태 시각화
                    angles = self.calculate_finger_angles(landmarks)
                    bent_count = 0
                    for finger in ['index', 'middle', 'ring', 'pinky']:
                        if finger in angles and angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                            bent_count += 1
                    
                    if bent_count >= 3:
                        # FIST READY 표시 (3개 이상 굽혀짐)
                        wrist = landmarks[self.WRIST]
                        wrist_pos = (int(wrist.x * w), int(wrist.y * h) - 40)
                        color = (0, 255, 0) if bent_count >= 3 else (0, 165, 255)
                        cv2.putText(frame_to_draw_on, f"FIST READY ({bent_count}/4)", wrist_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    elif bent_count > 0:
                        # 진행 상태 표시
                        wrist = landmarks[self.WRIST]
                        wrist_pos = (int(wrist.x * w), int(wrist.y * h) - 40)
                        cv2.putText(frame_to_draw_on, f"FIST ({bent_count}/4)", wrist_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    # 제스처 인식 및 전송
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
                        if position and raw_gesture_type.value == "flick":
                            debug_msg += f" @ {position.upper()}"
                        debug_msg += " ✓"
                        debug_messages_for_frame.append(debug_msg)
                    else:
                        stabilizer_stats = self.stabilizer.get_statistics()
                        pending_gesture = stabilizer_stats.get("current_gesture", "none")
                        
                        if pending_gesture != "none":
                            progress = stabilizer_stats.get("stability_progress", 0)
                            progress_percent = int(progress * 100)
                            confidence_avg = stabilizer_stats.get("confidence_avg", 0)
                            
                            # 제스처별 다른 유지시간 표시
                            if pending_gesture == "fist":
                                hold_time = Config.FIST_STABILITY_WINDOW
                            else:
                                hold_time = Config.DEFAULT_STABILITY_WINDOW
                                
                            debug_messages_for_frame.append(
                                f"{hand_label}: {pending_gesture} ({progress_percent}%/{hold_time}s) Conf: {confidence_avg:.2f}"
                            )
                            
                            # 프로그레스 바 그리기
                            if progress_percent > 0:
                                bar_x = 10 if hand_label == "Left" else 250
                                bar_y = 150
                                bar_width = 200
                                bar_height = 20
                                cv2.rectangle(frame_to_draw_on, (bar_x, bar_y), 
                                            (bar_x + bar_width, bar_y + bar_height), 
                                            (100, 100, 100), 2)
                                fill_width = int(bar_width * progress)
                                cv2.rectangle(frame_to_draw_on, (bar_x, bar_y), 
                                            (bar_x + fill_width, bar_y + bar_height), 
                                            (0, 255, 0), -1)
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
        
        logger.info("Ninja Master Hand Tracker (Dual-Gesture) initialized.")

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
        
        # 제스처 가이드
        guide_y = height - 200
        cv2.putText(frame, "== ENHANCED DUAL GESTURE MODE ==", (10, guide_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        guide_y += 30
        gestures_info = [
            ("FLICK", "Index-Mid Together + Fast Up (0.15s)", (0, 255, 0)),
            ("FIST", "Bend 3+ Fingers (Hold 0.4s)", (0, 200, 255))
        ]
        
        for i, (name, desc, color) in enumerate(gestures_info):
            y_pos = guide_y + (i * 25)
            cv2.putText(frame, f"{name}: {desc}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 임계값 정보
        info_y = guide_y + 60
        cv2.putText(frame, f"Flick: <{Config.FLICK_FINGER_DISTANCE_THRESHOLD:.3f}, >{Config.FLICK_SPEED_THRESHOLD}, Hold: {Config.DEFAULT_STABILITY_WINDOW}s", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Fist: 3+ fingers, <{Config.FIST_ANGLE_THRESHOLD}°, Hold: {Config.FIST_STABILITY_WINDOW}s", 
                    (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 하단 정보
        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def run(self):
        """메인 루프"""
        logger.info("Starting Ninja Master - ENHANCED DUAL GESTURE MODE...")
        logger.info("Improvements:")
        logger.info("- FLICK: Fast response (0.15s stability)")
        logger.info("- FIST: Slower recognition (0.4s stability)")
        logger.info("- Better FLICK detection (lower thresholds)")
        logger.info("- Improved FIST detection (3+ fingers, angle < 110°)")
        logger.info("- Smart conflict prevention")
        
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
                window_name = "Ninja Master - Differential Timing"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 360)
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
        from test_osc_communication import OSCTester
        
        logger.info("=== OSC 테스트 모드 (Enhanced Dual Gesture) ===")
        tester = OSCTester()
        tester.start_server()

        while True:
            print("\n테스트 옵션:")
            print("1. 모든 제스처 테스트")
            print("2. 위치별 FLICK 테스트")
            print("3. 종료")
            
            choice = input("선택: ")
            
            if choice == "1":
                gestures = [
                    ("flick", "throw_shuriken", 0.85, True),   # 위치 있음
                    ("fist", "activate_fist", 0.8, False)       # 위치 없음
                ]
                
                for gesture, action, confidence, has_position in gestures:
                    print(f"\n- {gesture.upper()}")
                    
                    if gesture == "flick":
                        print("  (빠른 반응 0.15초)")
                    elif gesture == "fist":
                        print("  (긴 유지시간 0.4초)")
                    
                    # OSC 메시지 전송
                    tester.client.send_message("/ninja/gesture/type", gesture)
                    tester.client.send_message("/ninja/gesture/action", action)
                    tester.client.send_message("/ninja/gesture/confidence", confidence)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    
                    if has_position:
                        tester.client.send_message("/ninja/gesture/position", "center")
                    
                    time.sleep(1.5)

            elif choice == "2":
                positions = ["left", "center", "right"]
                
                # FLICK만 위치 테스트
                for pos in positions:
                    print(f"\n=== FLICK @ {pos.upper()} ===")
                    
                    tester.client.send_message("/ninja/gesture/type", "flick")
                    tester.client.send_message("/ninja/gesture/action", "throw_shuriken")
                    tester.client.send_message("/ninja/gesture/confidence", 0.85)
                    tester.client.send_message("/ninja/gesture/hand", "Right")
                    tester.client.send_message("/ninja/gesture/position", pos)
                    time.sleep(1.0)

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
        # 개선된 설정
        custom_stabilizer_settings = {
            "stability_window": 0.15,       # 안정적이면서도 빠른 반응
            "confidence_threshold": 0.55,   # 더 관대한 임계값
            "cooldown_time": 0.25           # 적절한 쿨다운
        }
        
        print("\n" + "=" * 60)
        print("      닌자 마스터 - 제스처별 타이밍 모드")
        print("=" * 60)
        print("\n주요 개선사항:")
        print("  • 제스처별 다른 반응 시간:")
        print("    - FLICK: 0.15초 유지 (빠른 반응)")
        print("    - FIST: 0.4초 유지 (안정적 인식)")
        print("  • FLICK 인식율 향상:")
        print("    - 속도 임계값: 120 픽셀/초")
        print("    - 수직 비율: 40%")
        print("    - 손가락 거리: < 0.06")
        print("  • FIST 인식 개선:")
        print("    - 3개 이상 손가락 굽히면 인식")
        print("    - 각도 < 110° (더 관대함)")
        print("    - 더 긴 유지시간으로 안정성 확보")
        print("\n지원 제스처:")
        print("  • FLICK: 검지-중지 붙이고 위로 빠르게 (0.15초 유지)")
        print("  • FIST: 3개 이상 손가락 굽히기 (0.4초 유지)")
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