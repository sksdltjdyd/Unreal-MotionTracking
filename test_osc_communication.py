# test_osc_communication.py - OSC 통신 테스트 도구

import time
from pythonosc import udp_client, dispatcher, osc_server
import threading

class OSCTester:
    def __init__(self, server_ip="127.0.0.1", server_port=7000):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client = udp_client.SimpleUDPClient(server_ip, server_port)
        
        # 디스패처 설정
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/ninja/gesture/type", self.handle_gesture_type)
        self.dispatcher.map("/ninja/gesture/action", self.handle_gesture_action)
        self.dispatcher.map("/ninja/gesture/confidence", self.handle_confidence)
        self.dispatcher.map("/ninja/gesture/direction", self.handle_direction)
        self.dispatcher.map("/ninja/gesture/speed", self.handle_speed)
        self.dispatcher.map("/ninja/gesture/hand", self.handle_hand)
        self.dispatcher.map("/ninja/gesture/position", self.handle_position)
        self.dispatcher.map("/ninja/gesture/position_action", self.handle_position_action)
        self.dispatcher.map("/ninja/hand/detected", self.handle_hand_detected)
        self.dispatcher.map("/ninja/hand/count", self.handle_hand_count)
        
        self.server = None
        self.server_thread = None
        
    def handle_gesture_type(self, unused_addr, args):
        print(f"[GESTURE TYPE] {args}")
        
    def handle_gesture_action(self, unused_addr, args):
        print(f"[ACTION] {args}")
        
    def handle_confidence(self, unused_addr, args):
        print(f"[CONFIDENCE] {args:.2f}")
        
    def handle_direction(self, unused_addr, *args):
        if len(args) == 2:
            print(f"[DIRECTION] X: {args[0]:.2f}, Y: {args[1]:.2f}")
        else:
            print(f"[DIRECTION] {args}")
            
    def handle_speed(self, unused_addr, args):
        print(f"[SPEED] {args:.1f}")
        
    def handle_hand(self, unused_addr, args):
        print(f"[HAND] {args}")
        
    def handle_position(self, unused_addr, args):
        print(f"[POSITION] {args}")
        
    def handle_position_action(self, unused_addr, args):
        print(f"[POSITION ACTION] {args}")
        
    def handle_hand_detected(self, unused_addr, args):
        print(f"[HAND DETECTED] {'Yes' if args else 'No'}")
        
    def handle_hand_count(self, unused_addr, args):
        print(f"[HAND COUNT] {args}")
        
    def start_server(self):
        """OSC 서버 시작"""
        try:
            self.server = osc_server.ThreadingOSCUDPServer(
                (self.server_ip, self.server_port), self.dispatcher
            )
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"OSC 서버가 {self.server_ip}:{self.server_port}에서 시작되었습니다.")
        except Exception as e:
            print(f"서버 시작 오류: {e}")
            
    def stop_server(self):
        """OSC 서버 중지"""
        if self.server:
            self.server.shutdown()
            print("OSC 서버가 중지되었습니다.")
            
    def send_test_gesture(self, gesture_type, **kwargs):
        """테스트 제스처 전송"""
        print(f"\n--- {gesture_type.upper()} 제스처 전송 ---")
        
        # 기본값 설정
        action = kwargs.get('action', '')
        confidence = kwargs.get('confidence', 0.8)
        hand = kwargs.get('hand', 'Right')
        position = kwargs.get('position', None)
        direction = kwargs.get('direction', None)
        speed = kwargs.get('speed', None)
        
        # OSC 메시지 전송
        self.client.send_message("/ninja/gesture/type", gesture_type)
        
        if action:
            self.client.send_message("/ninja/gesture/action", action)
            
        self.client.send_message("/ninja/gesture/confidence", confidence)
        self.client.send_message("/ninja/gesture/hand", hand)
        
        if position:
            self.client.send_message("/ninja/gesture/position", position)
            self.client.send_message("/ninja/gesture/position_action", f"{action}_{position}")
            
        if direction:
            self.client.send_message("/ninja/gesture/direction", direction)
            
        if speed:
            self.client.send_message("/ninja/gesture/speed", speed)
            
        print(f"제스처 전송 완료: {gesture_type}")
        
if __name__ == "__main__":
    print("OSC 통신 테스터")
    print("=" * 40)
    
    tester = OSCTester()
    tester.start_server()
    
    try:
        while True:
            print("\n테스트 옵션:")
            print("1. FLICK 테스트 (위치별)")
            print("2. PALM PUSH 테스트 (위치별)")
            print("3. FIST 테스트")
            print("4. 손 감지 상태 테스트")
            print("5. 종료")
            
            choice = input("\n선택: ")
            
            if choice == "1":
                positions = ["left", "center", "right"]
                for pos in positions:
                    tester.send_test_gesture(
                        "flick",
                        action="throw_shuriken",
                        position=pos,
                        direction=[1.0, 0.0],
                        speed=300.0,
                        confidence=0.9
                    )
                    time.sleep(1)
                    
            elif choice == "2":
                positions = ["left", "center", "right"]
                for pos in positions:
                    tester.send_test_gesture(
                        "palm_push",
                        action="shock_wave",
                        position=pos,
                        confidence=0.85
                    )
                    time.sleep(1)
                    
            elif choice == "3":
                tester.send_test_gesture(
                    "fist",
                    action="block_attack",
                    confidence=0.8
                )
                
            elif choice == "4":
                print("\n손 감지 테스트")
                tester.client.send_message("/ninja/hand/detected", 1)
                tester.client.send_message("/ninja/hand/count", 2)
                time.sleep(0.5)
                tester.client.send_message("/ninja/hand/detected", 0)
                tester.client.send_message("/ninja/hand/count", 0)
                
            elif choice == "5":
                break
                
            else:
                print("잘못된 선택입니다.")
                
    except KeyboardInterrupt:
        print("\n\n중단됨.")
    finally:
        tester.stop_server()
        print("테스터 종료.")