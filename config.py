"""
Configuration file for Object Detection + Obstacle Avoidance Robot
Centralized settings for easy adjustment and calibration
"""

# === HARDWARE PIN CONFIGURATIONS ===

# L298N Motor Driver GPIO Pins
MOTOR_PINS = {
    'LEFT_PWM': 18,     # ENA - Left motor speed control
    'LEFT_IN1': 24,     # IN1 - Left motor direction 1  
    'LEFT_IN2': 23,     # IN2 - Left motor direction 2
    'RIGHT_PWM': 12,    # ENB - Right motor speed control
    'RIGHT_IN3': 25,    # IN3 - Right motor direction 1
    'RIGHT_IN4': 8,     # IN4 - Right motor direction 2
}

# HC-SR04 Ultrasonic Sensor GPIO Pins
SENSOR_PINS = {
    'FRONT_TRIG': 16,   # Front sensor trigger
    'FRONT_ECHO': 20,   # Front sensor echo
    'SIDE_TRIG': 21,    # Side sensor trigger
    'SIDE_ECHO': 26,    # Side sensor echo
}

# === DETECTION PARAMETERS ===

# YOLO Object Detection Settings
DETECTION_CONFIG = {
    'MODEL_NAME': 'yolov5s.pt',      # YOLOv5 model (s=small/fast, m=medium, l=large)
    'CONFIDENCE_THRESHOLD': 0.5,      # Minimum confidence for object detection
    'VISION_DISTANCE_THRESHOLD': 30,  # cm - Stop if object detected within this distance
    'FRONT_ZONE_WIDTH': 0.4,         # Width of front detection zone (0.0-1.0)
}

# Camera Settings
CAMERA_CONFIG = {
    'WIDTH': 640,                     # Camera frame width
    'HEIGHT': 480,                    # Camera frame height  
    'FPS': 30,                        # Frames per second
}

# === MOVEMENT PARAMETERS ===

# Motor Speed Settings (0-100)
MOTOR_SPEEDS = {
    'DEFAULT_SPEED': 70,              # Normal forward movement speed
    'TURN_SPEED': 60,                 # Speed during turns
    'BACKUP_SPEED': 50,               # Speed when backing up
}

# Movement Timing (seconds)
MOVEMENT_TIMING = {
    'TURN_DURATION': 1.2,             # How long to turn when avoiding objects
    'BACKUP_DURATION': 0.8,           # How long to back up during emergency stop
    'EMERGENCY_TURN_DURATION': 1.5,   # How long to turn during emergency avoidance
}

# === SENSOR PARAMETERS ===

# Distance Thresholds (centimeters)
DISTANCE_THRESHOLDS = {
    'ULTRASONIC_EMERGENCY': 20,       # Emergency stop distance for ultrasonic sensors
    'ULTRASONIC_WARNING': 30,         # Warning distance for ultrasonic sensors
    'VISION_OBSTACLE': 30,            # Stop distance for vision-detected objects
}

# Sensor Update Rates (Hz)
SENSOR_RATES = {
    'DISTANCE_UPDATE_RATE': 10,       # Ultrasonic sensor reading frequency
    'DETECTION_UPDATE_RATE': 10,      # Object detection frequency
}

# === SYSTEM PARAMETERS ===

# Main Loop Settings
SYSTEM_CONFIG = {
    'MAIN_LOOP_DELAY': 0.2,          # seconds - Main control loop frequency
    'STATUS_PRINT_INTERVAL': 25,      # Loop cycles between status prints
    'SENSOR_TIMEOUT': 0.03,          # seconds - Ultrasonic sensor timeout
}

# Threading Settings
THREAD_CONFIG = {
    'DAEMON_THREADS': True,           # Use daemon threads for background tasks
    'THREAD_STARTUP_DELAY': 0.5,     # seconds - Delay after starting threads
}

# === CALIBRATION PARAMETERS ===

# Distance Estimation Calibration
CALIBRATION = {
    'PIXEL_TO_CM_REFERENCE': 100,     # Reference pixel area for distance estimation
    'DISTANCE_CONVERSION_FACTOR': 10,  # Conversion factor for vision distance estimation
    'MIN_DETECTION_AREA': 500,       # Minimum pixel area to consider as valid detection
}

# Safety Margins
SAFETY_MARGINS = {
    'EMERGENCY_STOP_MARGIN': 5,       # cm - Additional margin for emergency stops
    'TURN_CLEARANCE_MARGIN': 10,     # cm - Minimum clearance after turns
}

# === DEBUG AND LOGGING ===

# Debug Settings
DEBUG_CONFIG = {
    'ENABLE_DISTANCE_LOGGING': True,  # Print distance measurements
    'ENABLE_DETECTION_LOGGING': True, # Print object detections
    'ENABLE_ACTION_LOGGING': True,    # Print robot actions
    'LOG_INTERVAL': 2,                # seconds - Interval between debug logs
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'MONITOR_FPS': False,             # Monitor detection frame rate
    'MONITOR_CPU_USAGE': False,       # Monitor CPU usage
    'PERFORMANCE_LOG_INTERVAL': 10,   # seconds
}

# === HARDWARE SPECIFIC SETTINGS ===

# PWM Configuration
PWM_CONFIG = {
    'FREQUENCY': 1000,                # Hz - PWM frequency for motor control
    'STARTUP_DUTY_CYCLE': 0,         # Initial PWM duty cycle
}

# GPIO Configuration  
GPIO_CONFIG = {
    'MODE': 'BCM',                    # GPIO numbering mode
    'WARNINGS': False,                # Disable GPIO warnings
    'CLEANUP_ON_EXIT': True,          # Clean up GPIO on program exit
}

# === BEHAVIOR TUNING ===

# Avoidance Behavior
BEHAVIOR_CONFIG = {
    'PREFER_RIGHT_TURNS': True,       # Prefer right turns for object avoidance
    'CONSECUTIVE_CLEAR_THRESHOLD': 5,  # Cycles of clear path before confident forward movement
    'STUCK_DETECTION_THRESHOLD': 10,  # Cycles of same action before considering stuck
}


def open_camera(indices=(0, 1, 2), width=None, height=None, fps=None, retries=2, wait=1.0):
    """
    Try to open a system camera robustly across platforms (macOS, Linux).

    Tries a list of device indices and attempts the macOS AVFoundation backend
    first (if available), then falls back to the default OpenCV backend. It
    validates by trying to read a single frame.

    Returns an opened cv2.VideoCapture object on success, or None on failure.
    """
    import cv2
    import time

    # Use config defaults when not provided
    if width is None:
        width = CAMERA_CONFIG.get('WIDTH', 640)
    if height is None:
        height = CAMERA_CONFIG.get('HEIGHT', 480)
    if fps is None:
        fps = CAMERA_CONFIG.get('FPS', 30)

    backends_to_try = []

    # Prefer AVFoundation on macOS if available in this build of OpenCV
    if hasattr(cv2, 'CAP_AVFOUNDATION'):
        backends_to_try.append(cv2.CAP_AVFOUNDATION)

    # Always allow default backend (0) as fallback
    backends_to_try.append(0)

    for backend in backends_to_try:
        for idx in indices:
            for attempt in range(retries):
                try:
                    if backend == 0:
                        cap = cv2.VideoCapture(idx)
                    else:
                        cap = cv2.VideoCapture(idx, backend)

                    if not cap or not cap.isOpened():
                        # brief wait before retrying
                        time.sleep(wait)
                        continue

                    # Set preferred properties (may be ignored by some cameras)
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                        cap.set(cv2.CAP_PROP_FPS, int(fps))
                    except Exception:
                        pass

                    # Validate by reading a single frame
                    ret, _ = cap.read()
                    if ret:
                        print(f"Camera opened (index={idx}, backend={'default' if backend==0 else backend})")
                        return cap
                    else:
                        cap.release()
                        time.sleep(wait)
                except Exception:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(wait)

    # If we reach here, no camera could be opened
    print("open_camera: no camera could be opened with tried backends/indices")
    return None


# Adaptive Behavior
ADAPTIVE_CONFIG = {
    'ENABLE_SPEED_ADAPTATION': False, # Adjust speed based on obstacles
    'ENABLE_DYNAMIC_THRESHOLDS': False, # Adjust thresholds based on environment
    'LEARNING_RATE': 0.1,            # Rate of adaptive changes
}
