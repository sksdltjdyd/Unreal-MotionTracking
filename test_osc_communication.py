# test_osc_communication.py - OSC 통신 테스트 도구

import time
from pythonosc import udp_client, osc_server, dispatcher
import threading
import random

class OSCTester:
    """OSC 통신 테스트를 위한 도구"""
    
    def __init__(self, send_port=7000, receive_port=7001):
        # 송신용 클라이언트
        self.client = udp_client.SimpleUDPClient("127.0.0.1", send_port)
        
        # 수신 테스트용 서버
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.set_default_handler(self.print_handler)
        
        # 특정 주소 핸들러 등록
        self.dispatcher.map("/ninja/gesture/type", self.gesture_type_handler)
        self.dispatcher.map("/ninja/gesture/direction", self.gesture_direction_handler)
        self.dispatcher.map("/ninja/gesture/speed", self.gesture_speed_handler)
        self.dispatcher.map("/ninja/hand/*", self.hand_handler)
        
        self.server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", receive_port), self.dispatcher)
        
        print(f"OSC 테스터 초기화")
        print(f"송신 포트: {send_port} (Python → Unreal)")
        print(f"수신 포트: {receive_port} (Unreal → Python)")
        print("-" * 50)
    
    def print_handler(self, address, *args):
        """기본 핸들러 - 모든 메시지 출력"""
        print(f"[OSC 수신] {address}: {args}")
    
    def gesture_type_handler(self, address, gesture_type):
        """제스처 타입 핸들러"""
        print(f"[제스처] 타입: {gesture_type}")
    
    def gesture_direction_handler(self, address, *args):
        """제스처 방향 핸들러"""
        if len(args) == 2:
            print(f"[제스처] 방향: ({args[0]:.2f}, {args[1]:.2f})")
        else:
            print(f"[제스처] 방향: {args[0]}")
    
    def gesture_speed_handler(self, address, speed):
        """제스처 속도 핸들러"""
        print(f"[제스처] 속도: {speed:.1f} pixels/sec")
    
    def hand_handler(self, address, *args):
        """손 관련 메시지 핸들러"""
        print(f"[손] {address}: {args}")
    
    def start_server(self):
        """수신 서버 시작"""
        print("OSC 수신 서버 시작...")
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
    
    def test_all_gestures(self):
        """모든 제스처 타입 테스트 전송"""
        print("\n=== 모든 제스처 테스트 ===")
        
        gestures = [
            {
                "type": "flick",
                "data": {
                    "direction": [0.7, 0.3],
                    "speed": 650.0,
                    "confidence": 0.9
                }
            },
            {
                "type": "fist",
                "data": {
                    "confidence": 0.95
                }
            },
            {
                "type": "palm_push",
                "data": {
                    "confidence": 0.85
                }
            },
            {
                "type": "circle",
                "data": {
                    "direction": "cw",
                    "confidence": 0.8
                }
            }
        ]
        
        for gesture in gestures:
            print(f"\n테스트: {gesture['type']}")
            
            # 제스처 타입
            self.client.send_message("/ninja/gesture/type", gesture['type'])
            
            # 신뢰도
            self.client.send_message("/ninja/gesture/confidence", 
                                   gesture['data']['confidence'])
            
            # 추가 데이터
            if 'direction' in gesture['data']:
                if isinstance(gesture['data']['direction'], list):
                    self.client.send_message("/ninja/gesture/direction", 
                                           gesture['data']['direction'])
                else:
                    self.client.send_message("/ninja/gesture/direction", 
                                           gesture['data']['direction'])
            
            if 'speed' in gesture['data']:
                self.client.send_message("/ninja/gesture/speed", 
                                       gesture['data']['speed'])
            
            time.sleep(1)
        
        print("\n모든 제스처 테스트 완료!")
    
    def test_hand_tracking(self):
        """손 추적 테스트"""
        print("\n=== 손 추적 테스트 ===")
        
        # 손 감지 상태
        print("1. 손 없음")
        self.client.send_message("/ninja/hand/detected", 0)
        self.client.send_message("/ninja/hand/count", 0)
        time.sleep(1)
        
        print("2. 한 손 감지")
        self.client.send_message("/ninja/hand/detected", 1)
        self.client.send_message("/ninja/hand/count", 1)
        time.sleep(1)
        
        print("3. 양손 감지")
        self.client.send_message("/ninja/hand/detected", 1)
        self.client.send_message("/ninja/hand/count", 2)
        
        # 랜드마크 테스트 (검지 끝)
        print("4. 검지 끝 좌표 전송")
        self.client.send_message("/ninja/landmark/8", [0.5, 0.3, -0.1])
        
        print("\n손 추적 테스트 완료!")
    
    def simulate_gameplay(self):
        """게임플레이 시뮬레이션"""
        print("\n=== 게임플레이 시뮬레이션 ===")
        
        scenarios = [
            {
                "name": "수리검 3연발",
                "duration": 3,
                "actions": [
                    {"type": "flick", "direction": [0.7, 0.0], "speed": 600, "delay": 0.3},
                    {"type": "flick", "direction": [0.5, 0.5], "speed": 550, "delay": 0.3},
                    {"type": "flick", "direction": [-0.7, 0.0], "speed": 620, "delay": 0.5},
                ]
            },
            {
                "name": "근접 공격 콤보",
                "duration": 2,
                "actions": [
                    {"type": "fist", "delay": 0.4},
                    {"type": "fist", "delay": 0.4},
                    {"type": "palm_push", "delay": 0.5},
                ]
            },
            {
                "name": "특수기 발동",
                "duration": 3,
                "actions": [
                    {"type": "circle", "direction": "cw", "delay": 1},
                    {"type": "circle", "direction": "ccw", "delay": 1},
                ]
            },
            {
                "name": "방어 후 반격",
                "duration": 4,
                "actions": [
                    {"type": "cross_block", "delay": 1.5},
                    {"type": "flick", "direction": [1.0, 0.0], "speed": 800, "delay": 0.5},
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n시나리오: {scenario['name']} ({scenario['duration']}초)")
            print("-" * 30)
            
            # 손 감지
            self.client.send_message("/ninja/hand/detected", 1)
            self.client.send_message("/ninja/hand/count", 1)
            
            for i, action in enumerate(scenario['actions']):
                print(f"  액션 {i+1}: {action['type']}", end="")
                
                # 제스처 전송
                self.client.send_message("/ninja/gesture/type", action['type'])
                self.client.send_message("/ninja/gesture/confidence", 0.9)
                self.client.send_message("/ninja/gesture/hand", "Right")
                
                # 추가 파라미터
                if 'direction' in action:
                    self.client.send_message("/ninja/gesture/direction", action['direction'])
                    if isinstance(action['direction'], list):
                        print(f" → 방향: {action['direction']}", end="")
                    else:
                        print(f" → {action['direction']}", end="")
                
                if 'speed' in action:
                    self.client.send_message("/ninja/gesture/speed", action['speed'])
                    print(f" (속도: {action['speed']})", end="")
                
                print()  # 줄바꿈
                time.sleep(action['delay'])
            
            # 시나리오 간 대기
            time.sleep(1)
        
        print("\n게임플레이 시뮬레이션 완료!")
    
    def generate_random_gestures(self, duration=30):
        """랜덤 제스처 생성"""
        print(f"\n=== 랜덤 제스처 생성 ({duration}초) ===")
        print("Ctrl+C로 중단 가능")
        
        gesture_types = ["flick", "fist", "palm_push", "circle"]
        start_time = time.time()
        gesture_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                # 랜덤 제스처 선택
                gesture = random.choice(gesture_types)
                hand = random.choice(["Left", "Right"])
                confidence = random.uniform(0.7, 1.0)
                
                # 기본 메시지
                self.client.send_message("/ninja/gesture/type", gesture)
                self.client.send_message("/ninja/gesture/confidence", confidence)
                self.client.send_message("/ninja/gesture/hand", hand)
                
                # 제스처별 추가 데이터
                if gesture == "flick":
                    direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
                    # 정규화
                    norm = (direction[0]**2 + direction[1]**2)**0.5
                    if norm > 0:
                        direction = [direction[0]/norm, direction[1]/norm]
                    speed = random.uniform(400, 800)
                    
                    self.client.send_message("/ninja/gesture/direction", direction)
                    self.client.send_message("/ninja/gesture/speed", speed)
                    
                elif gesture == "circle":
                    direction = random.choice(["cw", "ccw"])
                    self.client.send_message("/ninja/gesture/direction", direction)
                
                # 손 상태
                self.client.send_message("/ninja/hand/detected", 1)
                self.client.send_message("/ninja/hand/count", 1 if random.random() > 0.3 else 2)
                
                gesture_count += 1
                remaining = duration - (time.time() - start_time)
                print(f"\r생성된 제스처: {gesture_count} | 남은 시간: {remaining:.1f}초", end="")
                
                # 랜덤 대기
                time.sleep(random.uniform(0.3, 1.5))
                
        except KeyboardInterrupt:
            print("\n\n랜덤 생성 중단됨")
        
        print(f"\n\n총 {gesture_count}개의 제스처 생성 완료!")
    
    def manual_test(self):
        """수동 테스트 모드"""
        print("\n=== 수동 테스트 모드 ===")
        print("명령어: flick, fist, palm, circle, quit")
        print("예: flick 0.7 0.3 600")
        print("-" * 50)
        
        while True:
            try:
                command = input("\n명령> ").strip().lower().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                
                # 제스처 타입 매핑
                gesture_map = {
                    "flick": "flick",
                    "fist": "fist",
                    "palm": "palm_push",
                    "circle": "circle"
                }
                
                if command[0] in gesture_map:
                    gesture_type = gesture_map[command[0]]
                    
                    # 기본 메시지
                    self.client.send_message("/ninja/gesture/type", gesture_type)
                    self.client.send_message("/ninja/gesture/confidence", 0.9)
                    self.client.send_message("/ninja/gesture/hand", "Right")
                    
                    # 추가 파라미터 처리
                    if command[0] == "flick" and len(command) >= 4:
                        direction = [float(command[1]), float(command[2])]
                        speed = float(command[3])
                        self.client.send_message("/ninja/gesture/direction", direction)
                        self.client.send_message("/ninja/gesture/speed", speed)
                        print(f"전송: {gesture_type} 방향={direction} 속도={speed}")
                    
                    elif command[0] == "circle" and len(command) >= 2:
                        direction = command[1]
                        self.client.send_message("/ninja/gesture/direction", direction)
                        print(f"전송: {gesture_type} 방향={direction}")
                    
                    else:
                        print(f"전송: {gesture_type}")
                else:
                    print("알 수 없는 명령어")
                    
            except ValueError:
                print("잘못된 입력 형식")
            except Exception as e:
                print(f"오류: {e}")

if __name__ == "__main__":
    tester = OSCTester()
    tester.start_server()
    
    while True:
        print("\n=== OSC 테스트 메뉴 ===")
        print("1. 모든 제스처 테스트")
        print("2. 손 추적 테스트")
        print("3. 게임플레이 시뮬레이션")
        print("4. 랜덤 제스처 생성 (30초)")
        print("5. 수동 테스트 모드")
        print("6. 종료")
        print("-" * 25)
        
        choice = input("선택> ").strip()
        
        if choice == "1":
            tester.test_all_gestures()
        elif choice == "2":
            tester.test_hand_tracking()
        elif choice == "3":
            tester.simulate_gameplay()
        elif choice == "4":
            duration = input("생성 시간(초) [30]: ").strip()
            duration = int(duration) if duration else 30
            tester.generate_random_gestures(duration)
        elif choice == "5":
            tester.manual_test()
        elif choice == "6":
            print("\n테스트 종료")
            break
        else:
            print("잘못된 선택입니다.")
        
        input("\n계속하려면 Enter를 누르세요...")