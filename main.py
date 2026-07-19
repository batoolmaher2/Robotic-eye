"""
Main Control Loop for Object Detection + Obstacle Avoidance Robot

This is the main entry point that coordinates:
1. YOLO object detection from Pi Camera
2. Ultrasonic sensor distance monitoring  
3. Motor control for autonomous navigation

Behavior Logic:
- Priority 1: Ultrasonic sensor obstacle < 20cm → Emergency stop and turn left
- Priority 2: YOLO object detection in front within 30cm → Stop and turn right
- Priority 3: Path clear → Move forward

Hardware Requirements:
- Raspberry Pi 4
- Pi Camera Module
- 2x HC-SR04 Ultrasonic Sensors
- L298N Motor Driver
- DC Motors and chassis

Author: Robotics Project
Date: August 2025
"""

import time
import signal
import sys
import cv2
import numpy as np
from threading import Event, Thread
from detection import ObjectDetector
from config import open_camera, CAMERA_CONFIG

# Try to import movement, but allow graceful fallback for non-Raspberry Pi systems
try:
    from movement import RobotMovement
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False
    RobotMovement = None

class ObstacleAvoidanceRobot:
    def __init__(self):
        """
        Initialize the complete robot system
        """
        print("=" * 50)
        print("OBSTACLE AVOIDANCE ROBOT STARTING...")
        print("=" * 50)
        
        # Create exit event for clean shutdown
        self.exit_event = Event()
        
        # Check if running on hardware with GPIO support
        if not HAS_GPIO:
            print("\n⚠️  WARNING: GPIO/Motor hardware not available.")
            print("Running in VISION-ONLY DEMO MODE (detection without movement)")
            self.demo_mode = True
        else:
            self.demo_mode = False
        
        # Initialize detection system
        print("\n1. Initializing object detection system...")
        try:
            self.detector = ObjectDetector(
                model_name='yolov5s.pt',  # Fast model for real-time detection
                confidence_threshold=0.5
            )
            print("✓ Object detection system ready")
        except Exception as e:
            print(f"✗ Failed to initialize object detection: {e}")
            raise
        
        # Initialize movement system only if hardware available
        self.movement = None
        if HAS_GPIO:
            print("\n2. Initializing movement control system...")
            try:
                self.movement = RobotMovement()
                print("✓ Movement control system ready")
            except Exception as e:
                print(f"✗ Failed to initialize movement system: {e}")
                print("Continuing in vision-only mode...")
                self.demo_mode = True
        
        # Configuration parameters
        self.ULTRASONIC_THRESHOLD = 20.0  # cm - Emergency stop distance
        self.VISION_THRESHOLD = 30.0      # cm - YOLO detection distance
        self.LOOP_DELAY = 0.2            # seconds - Main loop frequency
        
        # State tracking
        self.last_action = "startup"
        self.consecutive_clear_count = 0
        
        if self.demo_mode:
            print("\n2. Skipping movement system (demo mode)")
        
        print("\n3. Starting monitoring threads...")
        
        # Start background threads
        self.detector.start_detection_thread()
        
        if self.movement:
            self.movement.start_distance_monitoring()
        
        print("✓ All available systems initialized successfully!")
        if self.demo_mode:
            print("Vision Detection Mode: Camera is monitoring for obstacles.")
        else:
            print("\nRobot is ready for autonomous operation.")
        print("Press Ctrl+C to stop the robot safely.\n")
        
        # Initialize camera for live preview
        self.preview_camera = None
        self.preview_enabled = False
        self.preview_thread = None
        self.latest_frame = None
        self.preview_running = False
    
    def signal_handler(self, signum, frame):
        """
        Handle Ctrl+C gracefully
        """
        print("\n\nShutdown signal received...")
        self.preview_running = False
        self.exit_event.set()
    
    def start_preview(self):
        """
        Start live camera preview in separate thread
        Shows detected objects with bounding boxes
        """
        try:
            self.preview_camera = open_camera(width=CAMERA_CONFIG.get('WIDTH', 640),
                                              height=CAMERA_CONFIG.get('HEIGHT', 480),
                                              fps=CAMERA_CONFIG.get('FPS', 30))
            if self.preview_camera and getattr(self.preview_camera, 'isOpened', lambda: False)():
                self.preview_enabled = True
                self.preview_running = True
                self.preview_thread = Thread(target=self._preview_loop, daemon=False)
                self.preview_thread.start()
                print("✓ Live camera preview started")
                print("  - A window titled '🤖 Robot Vision - Live Detection' should appear")
                print("  - Press 'q' or ESC in the window to close it")
                print("  - Robot will continue running in background\n")
            else:
                print("⚠️  Could not open preview camera - running in background mode only")
        except Exception as e:
            print(f"⚠️  Preview camera error: {e}")
    
    def _preview_loop(self):
        """
        Background thread for live camera preview with detection overlays
        """
        window_name = '🤖 Robot Vision - Live Detection'
        
        while self.preview_running and not self.exit_event.is_set():
            try:
                ret, frame = self.preview_camera.read()
                if not ret:
                    continue
                
                # Get latest detections
                detections = self.detector.get_latest_detections()
                
                # Draw detection boxes
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # Draw bounding box (red for potential obstacles, green otherwise)
                    color = (0, 0, 255) if class_name in ['person', 'chair', 'couch', 'car', 'bicycle'] else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add status text
                status = "ROBOT RUNNING - Press 'q' to quit"
                cv2.putText(frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add instruction text at bottom
                instruction = "Objects detected (boxes shown above)"
                cv2.putText(frame, instruction, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Create window and show frame
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)
                cv2.imshow(window_name, frame)
                
                # Check for quit key (with small wait to prevent high CPU)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.preview_running = False
                    break
                
            except Exception as e:
                print(f"Preview error: {e}")
                break
        
        # Cleanup
        try:
            if self.preview_camera:
                self.preview_camera.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
    
    def analyze_situation(self):
        """
        Analyze current sensor data and determine appropriate action
        
        Returns:
            tuple: (action_string, context_dict)
        """
        context = {}
        
        # Get YOLO detection data
        vision_obstacle = self.detector.check_front_obstacle(
            distance_threshold=self.VISION_THRESHOLD
        )
        
        # Get ultrasonic data only if hardware available
        if self.movement:
            front_distance, side_distance = self.movement.get_distances()
            is_close, sensor_name, closest_distance = self.movement.is_obstacle_close(
                threshold=self.ULTRASONIC_THRESHOLD
            )
            context['front_distance'] = front_distance
            context['side_distance'] = side_distance
            
            # Priority 1: Emergency stop for very close obstacles (ultrasonic)
            if is_close:
                return 'emergency_stop', {
                    'sensor': sensor_name,
                    'distance': closest_distance,
                    'front_distance': front_distance,
                    'side_distance': side_distance
                }
        else:
            context['front_distance'] = 0
            context['side_distance'] = 0
        
        # Priority 2: Avoid objects detected by vision (YOLO)
        if vision_obstacle:
            return 'avoid_object', {
                'detection_type': 'vision',
                'front_distance': context.get('front_distance', 0),
                'side_distance': context.get('side_distance', 0)
            }
        
        # Priority 3: Path is clear - move forward
        return 'move_forward', context
    
    def execute_action(self, action, context):
        """
        Execute the determined action
        
        Args:
            action (str): Action to perform
            context (dict): Additional context information
        """
        current_time = time.strftime("%H:%M:%S")
        
        if action == 'emergency_stop':
            if self.last_action != 'emergency_stop':
                print(f"\n[{current_time}] 🚨 EMERGENCY STOP!")
                print(f"Obstacle detected by {context['sensor']} sensor at {context['distance']:.1f}cm")
            
            if self.movement:
                self.movement.emergency_stop_and_avoid()
            self.consecutive_clear_count = 0
            
        elif action == 'avoid_object':
            if self.last_action != 'avoid_object':
                print(f"\n[{current_time}] 👁️  OBJECT DETECTED - Avoiding")
                print(f"Vision system detected obstacle in front")
            
            if self.movement:
                self.movement.stop()
                time.sleep(0.3)
                print("Turning right to avoid detected object...")
                self.movement.turn_right(duration=1.2)
            else:
                print("[Demo Mode] Would turn right to avoid object")
            
            self.consecutive_clear_count = 0
            
        elif action == 'move_forward':
            self.consecutive_clear_count += 1
            
            # Only print status periodically to avoid spam
            if (self.last_action != 'move_forward' or 
                self.consecutive_clear_count % 25 == 0):  # Every 5 seconds at 5Hz
                
                print(f"[{current_time}] ✅ Path clear - Moving forward "
                      f"(Front: {context.get('front_distance', 0):.1f}cm, "
                      f"Side: {context.get('side_distance', 0):.1f}cm)")
            
            if self.movement:
                self.movement.move_forward()
            
        else:
            # Fallback - stop robot
            if self.movement:
                self.movement.stop()
        
        self.last_action = action
    
    def run(self):
        """
        Main control loop with integrated camera display
        """
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🤖 ROBOT AUTONOMOUS MODE ACTIVE")
        print("Monitoring sensors and making navigation decisions...\n")
        
        # Open camera for display (not in separate thread to avoid macOS issues)
        print("Opening camera for live preview...")
        try:
            display_camera = open_camera(width=CAMERA_CONFIG.get('WIDTH', 640),
                                         height=CAMERA_CONFIG.get('HEIGHT', 480),
                                         fps=CAMERA_CONFIG.get('FPS', 30))
            if display_camera and getattr(display_camera, 'isOpened', lambda: False)():
                print("✓ Camera opened for preview\n")
                use_display = True
            else:
                print("⚠️  Camera preview not available, running in background mode\n")
                use_display = False
        except Exception as e:
            print(f"⚠️  Camera error: {e}, running in background mode\n")
            use_display = False
        
        try:
            loop_count = 0
            while not self.exit_event.is_set():
                loop_count += 1
                
                # Analyze current situation
                action, context = self.analyze_situation()
                
                # Execute appropriate action
                self.execute_action(action, context)
                
                # Display camera frame with detections (every other loop to reduce overhead)
                if use_display and loop_count % 2 == 0:
                    try:
                        ret, frame = display_camera.read()
                        if ret:
                            # Get latest detections
                            detections = self.detector.get_latest_detections()
                            
                            # Draw detection boxes
                            for detection in detections:
                                x1, y1, x2, y2 = detection['bbox']
                                class_name = detection['class_name']
                                confidence = detection['confidence']
                                
                                # Color code: red for obstacles, green for safe
                                color = (0, 0, 255) if class_name in ['person', 'chair', 'couch', 'car', 'bicycle'] else (0, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label
                                label = f"{class_name} ({confidence:.2f})"
                                cv2.putText(frame, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add status text
                            cv2.putText(frame, "ROBOT RUNNING - Press 'q' to quit", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            cv2.imshow('🤖 Robot Vision - Live Detection', frame)
                            
                            # Check for quit key
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:
                                self.exit_event.set()
                                break
                    except Exception as e:
                        print(f"Display error: {e}")
                        use_display = False
                
                # Control loop frequency
                time.sleep(self.LOOP_DELAY)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"\nUnexpected error in main loop: {e}")
        finally:
            # Close camera before shutdown
            if use_display and display_camera:
                try:
                    display_camera.release()
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self.shutdown()
    
    def shutdown(self):
        """
        Clean shutdown of all systems
        """
        print("\n" + "=" * 50)
        print("SHUTTING DOWN ROBOT SYSTEMS...")
        print("=" * 50)
        
        if self.movement:
            # Stop motors first for safety
            print("1. Stopping motors...")
            try:
                self.movement.stop()
                print("✓ Motors stopped")
            except Exception as e:
                print(f"✗ Error stopping motors: {e}")
            
            # Clean up movement system
            print("2. Cleaning up movement system...")
            try:
                self.movement.cleanup()
                print("✓ Movement system cleaned up")
            except Exception as e:
                print(f"✗ Error cleaning up movement: {e}")
        else:
            print("1. Movement system not initialized (skipped)")
        
        # Clean up detection system
        print("3. Cleaning up detection system...")
        try:
            self.detector.cleanup()
            print("✓ Detection system cleaned up")
        except Exception as e:
            print(f"✗ Error cleaning up detection: {e}")
        
        print("\n🤖 Robot shutdown complete. Goodbye!")

def main():
    """
    Main entry point
    """
    try:
        # Create and run robot
        robot = ObstacleAvoidanceRobot()
        robot.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("Please check hardware connections and try again.")
    finally:
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()
