# gesture_recognizer.py - 닌자 게임 인식 시스템 (v2 - 멀티 제스처, 양손 지원)

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
    """제스처 타입 정의"""
    NONE = "none"
    FLICK = "flick"        # 표창 던지기
    FIST = "fist"          # 주먹 쥐기
    PINCH = "pinch"        # 엄지와 검지 맞닿기


class Config:
    """설정값 관리"""
    # OSC 설정
    OSC_IP = "127.0.0.1"
    OSC_PORT = 7000

    # MediaPipe 설정
    MAX_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.8
    MIN_TRACKING_CONFIDENCE = 0.6
    MODEL_COMPLEXITY = 0

    # 카메라 설정
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 60
    CAMERA_BUFFER_SIZE = 1

    # 제스처 임계값
    # Flick (튕기기)
    FLICK_SPEED_THRESHOLD = 150             # 수평/수직 속도 임계값
    FLICK_UPWARD_VELOCITY_THRESHOLD = 80    # 위로 튕기는 속도 임계값 추가
    FLICK_FINGER_DISTANCE_THRESHOLD = 0.045 # 검지-중지 거리 임계값 (약간 완화)

    # Fist (주먹)
    FIST_ANGLE_THRESHOLD = 90               # 손가락이 90도 이하로 굽혀지면 주먹으로 인식
    FIST_CONFIDENCE_THRESHOLD = 0.85        # 주먹 인식 신뢰도

    # Pinch (핀치)
    PINCH_DISTANCE_THRESHOLD = 0.04         # 엄지-검지 거리 임계값

    # 안정화 설정
    DEFAULT_STABILITY_WINDOW = 0.05
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_COOLDOWN_TIME = 0.3

    # 스무딩 및 필터링
    SMOOTHING_BUFFER_SIZE = 2
    MOVEMENT_THRESHOLD = 5


class SimpleStabilizer:
    """단순화된 제스처 안정화 - 빠른 반응"""
    def __init__(self, **kwargs):
        self.stability_window = kwargs.get('stability_window', Config.DEFAULT_STABILITY_WINDOW)
        self.confidence_threshold = kwargs.get('confidence_threshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        self.cooldown_time = kwargs.get('cooldown_time', Config.DEFAULT_COOLDOWN_TIME)
        self.last_gesture_time = {}
        self.current_gesture = "none"
        self.current_gesture_start = 0
        self.gesture_buffer = deque(maxlen=5) # 안정성을 위해 버퍼 약간 증가
        self.last_sent_gesture_info = {}

    def should_send_gesture(self, gesture_type, confidence, hand_label, position=None):
        current_time = time.time()
        gesture_key = f"{gesture_type}_{hand_label}"

        if gesture_type == "none":
            self.gesture_buffer.append("none")
            # 버퍼가 모두 none이면 현재 제스처 리셋
            if all(g == "none" for g in self.gesture_buffer):
                self.current_gesture = "none"
            return False, None

        if confidence < self.confidence_threshold:
            return False, None

        self.gesture_buffer.append(gesture_type)
        
        # 버퍼 내에서 가장 빈번한 제스처를 현재 제스처로 간주
        if len(self.gesture_buffer) > 0:
            most_common_gesture = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        else:
            most_common_gesture = "none"

        # 새로운 제스처 시작 또는 변경
        if most_common_gesture != self.current_gesture:
            self.current_gesture = most_common_gesture
            self.current_gesture_start = current_time
            return False, None

        # 안정화 시간 체크
        if current_time - self.current_gesture_start < self.stability_window:
            return False, None
        
        # 쿨다운 체크
        if gesture_key in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_key] < self.cooldown_time:
                return False, None

        # 제스처 전송
        self.last_gesture_time[gesture_key] = current_time
        self.gesture_buffer.clear() # 전송 후 버퍼 초기화

        return True, {"confidence": confidence, "position": position}

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
    """닌자 게임 제스처 인식기 - 멀티 제스처, 양손 지원 버전"""

    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings=None):
        self.client = udp_client.SimpleUDPClient(osc_ip or Config.OSC_IP, osc_port or Config.OSC_PORT)
        
        stabilizer_settings = stabilizer_settings or {}
        self.stabilizers = {
            "Left": SimpleStabilizer(**stabilizer_settings),
            "Right": SimpleStabilizer(**stabilizer_settings)
        }
        logger.info(f"Stabilizers initialized with settings: {stabilizer_settings}")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
            model_complexity=Config.MODEL_COMPLEXITY
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 랜드마크 인덱스
        self.WRIST, self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP = 0, 4, 8, 12, 16, 20
        self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP = 6, 10, 14, 18
        self.THUMB_IP = 3

        # 상태 추적용 변수
        self.prev_landmarks = {"Left": None, "Right": None}
        self.prev_time = {"Left": time.time(), "Right": time.time()}
        self.hand_positions = {"Left": "center", "Right": "center"}

        logger.info(f"Ninja Gesture Recognizer (Multi-Gesture) initialized - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z]))

    def calculate_pixel_distance(self, p1_pixel, p2_pixel):
        return np.linalg.norm(np.array(p1_pixel) - np.array(p2_pixel))

    def calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
        v2 = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 180.0
        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def calculate_hand_position(self, landmarks):
        wrist_x = landmarks[self.WRIST].x
        if wrist_x < 0.33: return "left"
        if wrist_x > 0.66: return "right"
        return "center"

    # [개선] Flick 감지: 위로 튕기는 조건 추가
    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        if self.prev_landmarks[hand_label] is None:
            return False, 0.0, "center"

        current_time = time.time()
        dt = current_time - self.prev_time[hand_label]
        if dt == 0 or dt > 0.2: # 시간 간격이 너무 길면 무시
            return False, 0.0, "center"

        position = self.calculate_hand_position(current_landmarks)
        
        # 조건 1: 검지와 중지가 붙어있는지 확인
        finger_dist = self.calculate_distance(current_landmarks[self.INDEX_TIP], current_landmarks[self.MIDDLE_TIP])
        if finger_dist > Config.FLICK_FINGER_DISTANCE_THRESHOLD:
            return False, 0.0, position
        
        # 속도 계산 (검지와 중지 평균 위치 기준)
        curr_index_tip, prev_index_tip = current_landmarks[self.INDEX_TIP], self.prev_landmarks[hand_label][self.INDEX_TIP]
        curr_middle_tip, prev_middle_tip = current_landmarks[self.MIDDLE_TIP], self.prev_landmarks[hand_label][self.MIDDLE_TIP]

        curr_avg_pos = np.array([(curr_index_tip.x + curr_middle_tip.x) / 2 * img_width, (curr_index_tip.y + curr_middle_tip.y) / 2 * img_height])
        prev_avg_pos = np.array([(prev_index_tip.x + prev_middle_tip.x) / 2 * img_width, (prev_index_tip.y + prev_middle_tip.y) / 2 * img_height])
        
        # 픽셀 이동 거리 및 속도
        pixel_distance = self.calculate_pixel_distance(curr_avg_pos, prev_avg_pos)
        if pixel_distance < Config.MOVEMENT_THRESHOLD:
            return False, 0.0, position

        total_velocity = pixel_distance / dt
        # Y축 속도 (위로 갈수록 y값이 작아지므로, (prev - curr) 사용)
        velocity_y = (prev_avg_pos[1] - curr_avg_pos[1]) / dt

        # 조건 2: 전체 속도가 임계값을 넘거나, 위로 튕기는 속도가 임계값을 넘는 경우
        is_flick_motion = (total_velocity > Config.FLICK_SPEED_THRESHOLD) or (velocity_y > Config.FLICK_UPWARD_VELOCITY_THRESHOLD)

        if is_flick_motion:
            logger.info(f"[{hand_label}] Flick Detected! Total_V: {total_velocity:.1f}, Upward_V: {velocity_y:.1f}")
            return True, total_velocity, position
            
        return False, 0.0, position

    # [신규] Fist(주먹) 감지
    def detect_fist(self, landmarks):
        try:
            # 모든 손가락이 굽혀졌는지 확인 (엄지 제외)
            bent_fingers = 0
            finger_indices = {
                'index': (self.INDEX_PIP, self.INDEX_TIP, self.WRIST),
                'middle': (self.MIDDLE_PIP, self.MIDDLE_TIP, self.WRIST),
                'ring': (self.RING_PIP, self.RING_TIP, self.WRIST),
                'pinky': (self.PINKY_PIP, self.PINKY_TIP, self.WRIST)
            }
            for finger, (pip, tip, wrist) in finger_indices.items():
                angle = self.calculate_angle(landmarks[wrist], landmarks[pip], landmarks[tip])
                if angle < Config.FIST_ANGLE_THRESHOLD:
                    bent_fingers += 1

            # 4개 손가락 모두 굽혀졌으면 Fist로 판단
            if bent_fingers >= 4:
                return True, Config.FIST_CONFIDENCE_THRESHOLD
            return False, 0.0
        except Exception:
            return False, 0.0

    # [신규] Pinch(핀치) 감지
    def detect_pinch(self, landmarks):
        try:
            # 엄지와 검지 끝 사이의 거리 확인
            distance = self.calculate_distance(landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP])
            if distance < Config.PINCH_DISTANCE_THRESHOLD:
                confidence = 1.0 - (distance / Config.PINCH_DISTANCE_THRESHOLD)
                return True, max(0.7, confidence) # 최소 신뢰도 보장
            return False, 0.0
        except Exception:
            return False, 0.0

    # [개선] 제스처 인식 로직: Flick 양손 지원 및 신규 제스처 추가
    def recognize_gesture(self, hand_landmarks_obj, hand_label, img_shape):
        landmarks = hand_landmarks_obj.landmark
        height, width = img_shape[:2]
        
        # 제스처 우선순위: Flick > Fist > Pinch
        
        # 1. Flick 감지 (양손 모두)
        is_flick, flick_speed, flick_position = self.detect_flick(landmarks, hand_label, width, height)
        if is_flick:
            confidence = min(0.7 + flick_speed / 500, 1.0)
            return GestureType.FLICK, {
                "action": "throw_shuriken", "speed": flick_speed,
                "confidence": confidence, "position": flick_position
            }
            
        # 2. Fist 감지
        is_fist, fist_confidence = self.detect_fist(landmarks)
        if is_fist:
            position = self.calculate_hand_position(landmarks)
            return GestureType.FIST, {
                "action": "charge_power", "confidence": fist_confidence, "position": position
            }

        # 3. Pinch 감지
        is_pinch, pinch_confidence = self.detect_pinch(landmarks)
        if is_pinch:
            position = self.calculate_hand_position(landmarks)
            return GestureType.PINCH, {
                "action": "select_item", "confidence": pinch_confidence, "position": position
            }

        # 모든 제스처에 해당하지 않으면 NONE
        return GestureType.NONE, {"confidence": 0.0}


    def send_gesture_osc(self, gesture_type_enum, gesture_data, hand_label):
        try:
            gesture_type_str = gesture_type_enum.value
            confidence = gesture_data.get("confidence", 0.0)
            action = gesture_data.get("action", "")
            position = gesture_data.get("position", "center")

            bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            
            # 기본 정보
            bundle_builder.add_content(osc_message_builder.OscMessageBuilder(address="/ninja/gesture/type").add_arg(gesture_type_str).build())
            bundle_builder.add_content(osc_message_builder.OscMessageBuilder(address="/ninja/gesture/action").add_arg(action).build())
            bundle_builder.add_content(osc_message_builder.OscMessageBuilder(address="/ninja/gesture/confidence").add_arg(float(confidence)).build())
            bundle_builder.add_content(osc_message_builder.OscMessageBuilder(address="/ninja/gesture/hand").add_arg(hand_label).build())
            bundle_builder.add_content(osc_message_builder.OscMessageBuilder(address="/ninja/gesture/position").add_arg(position).build())

            self.client.send(bundle_builder.build())
            
            log_msg = f"OSC Sent: {hand_label} {gesture_type_str.upper()} ({action}) @ {position.upper()} (Conf: {confidence:.2f})"
            logger.info(log_msg)

        except Exception as e:
            logger.error(f"OSC 전송 중 오류 발생: {e}")
            
    def process_frame(self, frame_input):
        frame_to_draw_on = frame_input.copy()
        debug_messages_for_frame = []

        try:
            rgb_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # 손이 감지되지 않으면 이전 랜드마크 초기화
            detected_hands = []
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks_obj in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[hand_idx].classification[0].label
                    detected_hands.append(hand_label)
                    
                    self.mp_drawing.draw_landmarks(
                        frame_to_draw_on, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 제스처 인식 및 안정화 처리
                    raw_gesture, raw_data = self.recognize_gesture(hand_landmarks_obj, hand_label, frame_input.shape)
                    
                    stabilizer = self.stabilizers[hand_label]
                    should_send, stabilized_data = stabilizer.should_send_gesture(
                        raw_gesture.value, raw_data.get("confidence", 0.0), hand_label, raw_data.get("position")
                    )
                    
                    if should_send:
                        self.send_gesture_osc(raw_gesture, raw_data, hand_label)
                        debug_messages_for_frame.append(f"[{hand_label}] {raw_gesture.value.upper()} @ {raw_data.get('position', 'N/A').upper()} ✓")
                    
                    # 현재 인식 시도 중인 제스처 표시
                    stats = stabilizer.get_statistics()
                    pending_gesture = stats.get("current_gesture", "none")
                    if pending_gesture != "none" and not should_send:
                        progress = min(stats.get("stability_progress", 0) / stabilizer.stability_window * 100, 100)
                        debug_messages_for_frame.append(f"[{hand_label}] {pending_gesture.upper()}... ({progress:.0f}%)")

                    # 이전 프레임 정보 업데이트
                    self.prev_landmarks[hand_label] = hand_landmarks_obj.landmark
                    self.prev_time[hand_label] = time.time()
            
            # 감지되지 않은 손의 정보는 리셋
            for hand_label in ["Left", "Right"]:
                if hand_label not in detected_hands:
                    self.prev_landmarks[hand_label] = None
                    self.stabilizers[hand_label].reset_if_idle()

        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
            traceback.print_exc()

        return frame_to_draw_on, debug_messages_for_frame

    def cleanup(self):
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        logger.info("Gesture recognizer cleaned up.")


class NinjaMasterHandTracker:
    """메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None, stabilizer_settings_override=None):
        self.gesture_recognizer = NinjaGestureRecognizer(
            osc_ip=osc_ip, osc_port=osc_port, stabilizer_settings=stabilizer_settings_override
        )
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # DSHOW 백엔드 사용 시도
        if not self.cap.isOpened():
            raise IOError("웹캠을 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.debug_mode = True
        
        logger.info("Ninja Master Hand Tracker (Multi-Gesture) initialized.")

    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_time
        if time_diff > 0:
            self.fps_counter.append(1.0 / time_diff)
        self.last_time = current_time
        return np.mean(self.fps_counter) if self.fps_counter else 0.0

    # [개선] 디버그 정보: 신규 제스처 가이드 추가
    def draw_debug_info(self, frame, fps, debug_messages_list):
        height, width = frame.shape[:2]
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 위치 가이드라인
        left_line = int(width * 0.33)
        right_line = int(width * 0.66)
        cv2.line(frame, (left_line, 0), (left_line, height), (100, 100, 100), 1)
        cv2.line(frame, (right_line, 0), (right_line, height), (100, 100, 100), 1)
        
        # 제스처 상태
        y_offset = 70
        for message in debug_messages_list:
            color = (0, 255, 255) if "✓" in message else (255, 255, 0)
            cv2.putText(frame, message, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y_offset += 30
            
        # 제스처 가이드
        guide_y = height - 120
        cv2.putText(frame, "== GESTURES (L/R HAND) ==", (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        guide_y += 25
        cv2.putText(frame, "FLICK: Index-Mid Close + Fast Move (Upward OK)", (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        guide_y += 20
        cv2.putText(frame, "FIST: Clench hand", (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
        guide_y += 20
        cv2.putText(frame, "PINCH: Touch Thumb and Index tips", (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "Q: Quit | D: Debug Toggle", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        logger.info("Starting Ninja Master - Multi-Gesture Mode...")
        logger.info("Recognizing: FLICK, FIST, PINCH for BOTH hands.")
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    logger.error("웹캠에서 프레임을 읽을 수 없습니다.")
                    break
                
                flipped_frame = cv2.flip(frame, 1)
                processed_frame, debug_messages = self.gesture_recognizer.process_frame(flipped_frame)
                current_fps = self.calculate_fps()
                
                if self.debug_mode:
                    self.draw_debug_info(processed_frame, current_fps, debug_messages)
                
                window_name = "Ninja Master - Multi-Gesture Recognizer"
                cv2.imshow(window_name, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("종료합니다.")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("리소스 정리 중...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'gesture_recognizer'):
            self.gesture_recognizer.cleanup()
        logger.info("프로그램 종료.")


if __name__ == "__main__":
    # 고속 반응을 위한 안정화 설정 오버라이드
    custom_stabilizer_settings = {
        "stability_window": 0.05,    # 50ms 유지 시 인식
        "confidence_threshold": 0.6, # 60% 신뢰도
        "cooldown_time": 0.3         # 300ms 쿨다운
    }
    
    print("\n" + "=" * 60)
    print("      닌자 마스터 - 멀티 제스처 고속 인식 시스템 v2.0")
    print("=" * 60)
    print("\n[기능]")
    print("  • 양손 지원: FLICK, FIST, PINCH 제스처")
    print("  • FLICK: 기존 방식 + 위로 튕기는 동작 추가")
    print("  • FIST: 주먹 쥔 동작")
    print("  • PINCH: 엄지-검지 맞댄 동작")
    print("\n[조작법]")
    print("  • Q - 종료")
    print("  • D - 디버그 정보 ON/OFF")
    print("=" * 60 + "\n")
    
    try:
        tracker = NinjaMasterHandTracker(
            stabilizer_settings_override=custom_stabilizer_settings
        )
        tracker.run()
    except Exception as e:
        logger.critical(f"프로그램 실행 중 치명적 오류 발생: {e}")
        traceback.print_exc()