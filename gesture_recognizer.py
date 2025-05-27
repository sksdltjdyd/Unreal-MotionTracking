# gesture_recognizer.py - 닌자 게임 인식 시스템 (완전판)

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
    CIRCLE_STD_THRESHOLD = 30    # pixels
    CIRCLE_MIN_POINTS = 15
    
    # 스무딩 설정
    SMOOTHING_BUFFER_SIZE = 3
    FPS_BUFFER_SIZE = 30

class GestureValidator:
    """제스처 유효성 검증"""
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
    """닌자 게임임 제스처 인식기"""
    
    def __init__(self, osc_ip=None, osc_port=None):
        # OSC 설정
        self.client = udp_client.SimpleUDPClient(
            osc_ip or Config.OSC_IP, 
            osc_port or Config.OSC_PORT
        )
        
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
        
        # 랜드마크 인덱스
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # 제스처 검증기
        self.validator = GestureValidator()
        
        # 상태 추적
        self.prev_landmarks = {"Left": None, "Right": None}
        self.prev_time = time.time()
        self.position_history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}
        self.circle_points = {"Left": deque(maxlen=20), "Right": deque(maxlen=20)}
        
        logger.info(f"닌자 제스처 인식기 초기화 완료 - OSC: {osc_ip or Config.OSC_IP}:{osc_port or Config.OSC_PORT}")
    
    def calculate_distance(self, p1, p2):
        """두 점 사이의 거리 계산"""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def calculate_angle(self, p1, p2, p3):
        """세 점으로 이루어진 각도 계산"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_finger_angles(self, landmarks):
        """손가락 굴곡 각도 계산"""
        angles = {}
        
        finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger, joints in finger_joints.items():
            if len(joints) >= 3:
                p1 = [landmarks[joints[0]].x, landmarks[joints[0]].y]
                p2 = [landmarks[joints[1]].x, landmarks[joints[1]].y]
                p3 = [landmarks[joints[2]].x, landmarks[joints[2]].y]
                
                angles[finger] = self.calculate_angle(p1, p2, p3)
        
        return angles
    
    def detect_fist(self, landmarks):
        """주먹 쥐기 감지"""
        angles = self.calculate_finger_angles(landmarks)
        
        closed_fingers = 0
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles and angles[finger] < Config.FIST_ANGLE_THRESHOLD:
                closed_fingers += 1
        
        confidence = closed_fingers / 4.0
        return closed_fingers >= 3, confidence
    
    def detect_flick(self, current_landmarks, hand_label, img_width, img_height):
        """손가락 튕기기 감지"""
        if self.prev_landmarks[hand_label] is None:
            return False, None, 0
        
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt == 0:
            return False, None, 0
        
        # 검지 끝 속도 계산
        curr_index = current_landmarks[self.INDEX_TIP]
        prev_index = self.prev_landmarks[hand_label][self.INDEX_TIP]
        
        curr_pos = np.array([curr_index.x * img_width, curr_index.y * img_height])
        prev_pos = np.array([prev_index.x * img_width, prev_index.y * img_height])
        
        velocity = self.calculate_distance(curr_pos, prev_pos) / dt
        
        if velocity > Config.FLICK_SPEED_THRESHOLD:
            direction = curr_pos - prev_pos
            direction_normalized = direction / np.linalg.norm(direction)
            
            # 검지가 펴져있는지 확인
            angles = self.calculate_finger_angles(current_landmarks)
            if 'index' in angles and angles['index'] > 120:
                return True, direction_normalized.tolist(), velocity
        
        return False, None, 0
    
    def detect_palm_push(self, landmarks, hand_label):
        """손바닥 밀기 감지"""
        angles = self.calculate_finger_angles(landmarks)
        
        # 손가락이 펴져있는지 확인
        extended_fingers = 0
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if finger in angles and angles[finger] > Config.PALM_EXTEND_THRESHOLD:
                extended_fingers += 1
        
        if extended_fingers >= 3:
            # z축 차이로 밀기 감지
            palm_center = landmarks[9]
            wrist = landmarks[self.WRIST]
            
            if abs(palm_center.z - wrist.z) > 0.1:
                confidence = min(abs(palm_center.z - wrist.z) * 5, 1.0)
                return True, confidence
        
        return False, 0
    
    def detect_circle(self, landmarks, hand_label, img_width, img_height):
        """원 그리기 감지"""
        index_tip = landmarks[self.INDEX_TIP]
        pos = np.array([index_tip.x * img_width, index_tip.y * img_height])
        
        self.circle_points[hand_label].append(pos)
        
        if len(self.circle_points[hand_label]) >= Config.CIRCLE_MIN_POINTS:
            points = np.array(self.circle_points[hand_label])
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            
            if np.std(distances) < Config.CIRCLE_STD_THRESHOLD:
                # 방향 판정
                angles = []
                for i in range(len(points) - 1):
                    v1 = points[i] - center
                    v2 = points[i + 1] - center
                    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                    angles.append(angle)
                
                avg_angle = np.mean(angles)
                direction = "cw" if avg_angle > 0 else "ccw"
                
                self.circle_points[hand_label].clear()
                return True, direction
        
        return False, None
    
    def recognize_gesture(self, hand_landmarks, hand_label, img_shape):
        """통합 제스처 인식"""
        height, width = img_shape[:2]
        current_gesture = GestureType.NONE
        gesture_data = {}
        
        # 1. 플릭 감지
        is_flick, direction, speed = self.detect_flick(
            hand_landmarks.landmark, hand_label, width, height
        )
        if is_flick:
            current_gesture = GestureType.FLICK
            gesture_data = {
                "direction": direction,
                "speed": speed,
                "confidence": 0.9
            }
        
        # 2. 주먹 감지
        else:
            is_fist, fist_confidence = self.detect_fist(hand_landmarks.landmark)
            if is_fist:
                current_gesture = GestureType.FIST
                gesture_data = {"confidence": fist_confidence}
        
        # 3. 손바닥 밀기 감지
        if current_gesture == GestureType.NONE:
            is_push, push_confidence = self.detect_palm_push(
                hand_landmarks.landmark, hand_label
            )
            if is_push:
                current_gesture = GestureType.PALM_PUSH
                gesture_data = {"confidence": push_confidence}
        
        # 4. 원 그리기 감지
        is_circle, circle_dir = self.detect_circle(
            hand_landmarks.landmark, hand_label, width, height
        )
        if is_circle:
            current_gesture = GestureType.CIRCLE
            gesture_data = {
                "direction": circle_dir,
                "confidence": 0.8
            }
        
        # 이전 프레임 저장
        self.prev_landmarks[hand_label] = hand_landmarks.landmark
        
        return current_gesture, gesture_data
    
    def send_gesture_osc(self, gesture_type, gesture_data, hand_label):
        """제스처 정보를 OSC로 전송"""
        try:
            # 유효성 검증
            confidence = gesture_data.get("confidence", 0.0)
            if not self.validator.validate(gesture_type.value, confidence, hand_label):
                return
            
            # OSC 번들 생성
            bundle = osc_bundle_builder.OscBundleBuilder(
                osc_bundle_builder.IMMEDIATELY
            )
            
            # 제스처 타입
            msg = osc_message_builder.OscMessageBuilder(
                address="/ninja/gesture/type"
            )
            msg.add_arg(gesture_type.value)
            bundle.add_content(msg.build())
            
            # 신뢰도
            msg = osc_message_builder.OscMessageBuilder(
                address="/ninja/gesture/confidence"
            )
            msg.add_arg(float(confidence))
            bundle.add_content(msg.build())
            
            # 방향
            if "direction" in gesture_data:
                msg = osc_message_builder.OscMessageBuilder(
                    address="/ninja/gesture/direction"
                )
                if isinstance(gesture_data["direction"], list):
                    msg.add_arg(float(gesture_data["direction"][0]))
                    msg.add_arg(float(gesture_data["direction"][1]))
                else:
                    msg.add_arg(str(gesture_data["direction"]))
                bundle.add_content(msg.build())
            
            # 속도
            if "speed" in gesture_data:
                msg = osc_message_builder.OscMessageBuilder(
                    address="/ninja/gesture/speed"
                )
                msg.add_arg(float(gesture_data["speed"]))
                bundle.add_content(msg.build())
            
            # 손 구분
            msg = osc_message_builder.OscMessageBuilder(
                address="/ninja/gesture/hand"
            )
            msg.add_arg(hand_label)
            bundle.add_content(msg.build())
            
            # 전송
            self.client.send(bundle.build())
            
            logger.debug(f"Sent gesture: {gesture_type.value} ({hand_label})")
            
        except Exception as e:
            logger.error(f"OSC 전송 오류: {e}")
    
    def send_hand_state(self, hand_count):
        """손 감지 상태 전송"""
        try:
            self.client.send_message("/ninja/hand/detected", 1 if hand_count > 0 else 0)
            self.client.send_message("/ninja/hand/count", hand_count)
        except Exception as e:
            logger.error(f"손 상태 전송 오류: {e}")
    
    def process_frame(self, frame):
        """프레임 처리 및 제스처 인식"""
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        debug_info = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                # 손 그리기
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # 손 구분
                hand_label = handedness.classification[0].label
                
                # 제스처 인식
                gesture_type, gesture_data = self.recognize_gesture(
                    hand_landmarks, hand_label, frame.shape
                )
                
                # OSC 전송
                if gesture_type != GestureType.NONE:
                    self.send_gesture_osc(gesture_type, gesture_data, hand_label)
                    debug_info.append(f"{hand_label}: {gesture_type.value}")
        
        # 손 감지 상태 전송
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        self.send_hand_state(hand_count)
        
        # 시간 업데이트
        self.prev_time = time.time()
        
        return frame, debug_info
    
    def cleanup(self):
        """리소스 정리"""
        self.hands.close()

class NinjaMasterHandTracker:
    """닌자 마스터 메인 트래커"""
    
    def __init__(self, osc_ip=None, osc_port=None):
        self.gesture_recognizer = NinjaGestureRecognizer(osc_ip, osc_port)
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.CAMERA_BUFFER_SIZE)
        
        # FPS 계산
        self.fps_counter = deque(maxlen=Config.FPS_BUFFER_SIZE)
        self.last_time = time.time()
        
        # 디버그 모드
        self.debug_mode = True
        
        logger.info("닌자 마스터 핸드 트래커 초기화 완료")
    
    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        if self.last_time > 0:
            fps = 1 / (current_time - self.last_time)
            self.fps_counter.append(fps)
        self.last_time = current_time
        
        if len(self.fps_counter) > 0:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0
    
    def draw_debug_info(self, frame, fps, debug_info):
        """디버그 정보 그리기"""
        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 제스처 정보
        y_offset = 70
        for info in debug_info:
            cv2.putText(frame, info, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y_offset += 30
        
        # 안내 문구
        cv2.putText(frame, "Ninja Master - Gesture Recognition", (10, Config.CAMERA_HEIGHT - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug", (10, Config.CAMERA_HEIGHT - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run(self):
        """메인 실행 루프"""
        logger.info("닌자 마스터 제스처 인식 시작... 'q'를 눌러 종료")
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    logger.error("웹캠에서 프레임을 읽을 수 없습니다.")
                    break
                
                # 좌우 반전 (거울 효과)
                frame = cv2.flip(frame, 1)
                
                # 제스처 인식 처리
                processed_frame, debug_info = self.gesture_recognizer.process_frame(frame)
                
                # FPS 계산
                fps = self.calculate_fps()
                
                # 디버그 정보 표시
                if self.debug_mode:
                    self.draw_debug_info(processed_frame, fps, debug_info)
                
                # 화면 표시
                cv2.imshow("Ninja Master", processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    logger.info(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        except Exception as e:
            logger.error(f"실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("프로그램 종료 중...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.gesture_recognizer.cleanup()
        logger.info("닌자 마스터 제스처 인식 종료")

# 테스트 모드를 위한 함수
def test_mode():
    """OSC 통신 테스트 모드"""
    from test_osc_communication import OSCTester
    
    print("=== OSC 통신 테스트 모드 ===")
    tester = OSCTester()
    tester.start_server()
    
    while True:
        print("\n테스트 옵션:")
        print("1. 모든 제스처 테스트")
        print("2. 손 추적 테스트")
        print("3. 게임플레이 시뮬레이션")
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

if __name__ == "__main__":
    import sys
    
    # 테스트 모드 체크
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mode()
    else:
        # 일반 실행
        try:
            tracker = NinjaMasterHandTracker()
            tracker.run()
        except KeyboardInterrupt:
            print("\n프로그램이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()