"""Comprehensive FastAPI server for complete hexapod robot control with advanced features."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
import time
import math
import numpy as np
from enum import Enum
import json
from datetime import datetime
import io

from servo import Servo
from led import Led
from buzzer import Buzzer
from ultrasonic import Ultrasonic
from adc import ADC
from control import Control
from imu import IMU
from camera import Camera

app = FastAPI(
    title="Hexapod Robot Control API",
    description="Comprehensive API for controlling all aspects of the hexapod robot",
    version="3.0.0"
)

# Initialize controllers
servo_controller = Servo()
_led = None
buzzer = Buzzer()
ultrasonic = Ultrasonic()
adc = ADC()
control = Control()
imu = IMU()
camera = Camera()

# Global state management
servo_state = {i: 90 for i in range(32)}
head_state = {"pan": 90, "tilt": 90}  # Track head position
movement_task = None
is_moving = False
telemetry_clients = []
config = {}

# Leg and servo mapping
LEG_MAP = {
    1: [15, 14, 13],  # Right front
    2: [12, 11, 10],  # Right middle
    3: [9, 8, 31],    # Right back
    4: [22, 23, 27],  # Left back
    5: [19, 20, 21],  # Left middle
    6: [16, 17, 18],  # Left front
}

# Joint names for clarity
JOINT_NAMES = {
    "coxa": 0,    # Hip joint
    "femur": 1,   # Upper leg
    "tibia": 2    # Lower leg
}

# --- Enums ---

class GaitType(str, Enum):
    TRIPOD = "tripod"
    WAVE = "wave"
    RIPPLE = "ripple"
    CUSTOM = "custom"

class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"

class LedMode(str, Enum):
    OFF = "off"
    SOLID = "solid"
    PULSE = "pulse"
    RAINBOW = "rainbow"
    THEATER = "theater"
    CUSTOM = "custom"

class CameraMode(str, Enum):
    OFF = "off"
    PREVIEW = "preview"
    STREAM = "stream"
    CAPTURE = "capture"

# --- Pydantic Models ---

class ServoPosition(BaseModel):
    channel: int = Field(..., ge=0, le=31)
    angle: int = Field(..., ge=0, le=180)
    duration: float = Field(1.0, ge=0.1, le=10.0, description="Duration in seconds")
    
class HeadPosition(BaseModel):
    pan: Optional[int] = Field(None, ge=0, le=180, description="Pan angle (left/right)")
    tilt: Optional[int] = Field(None, ge=50, le=180, description="Tilt angle (up/down)")
    duration: float = Field(1.0, ge=0.1, le=10.0, description="Duration in seconds")

class MultiServoPosition(BaseModel):
    positions: List[ServoPosition]
    synchronized: bool = Field(True, description="Move all servos simultaneously")

class LegPosition(BaseModel):
    x: float = Field(..., description="X coordinate in mm")
    y: float = Field(..., description="Y coordinate in mm")
    z: float = Field(..., description="Z coordinate in mm")
    duration: float = Field(1.0, ge=0.1, le=10.0)

class LegAngles(BaseModel):
    coxa: int = Field(..., ge=0, le=180)
    femur: int = Field(..., ge=0, le=180)
    tibia: int = Field(..., ge=0, le=180)
    duration: float = Field(1.0, ge=0.1, le=10.0)

class BodyPose(BaseModel):
    height: float = Field(default=-90, ge=-50, le=50, description="Body height adjustment")
    roll: float = Field(default=0, ge=-15, le=15, description="Roll angle in degrees")
    pitch: float = Field(default=0, ge=-15, le=15, description="Pitch angle in degrees")
    yaw: float = Field(default=0, ge=-15, le=15, description="Yaw angle in degrees")
    x_offset: float = Field(default=0, ge=-40, le=40, description="X position offset")
    y_offset: float = Field(default=0, ge=-40, le=40, description="Y position offset")
    duration: float = Field(1.0, ge=0.1, le=10.0)

class GaitParameters(BaseModel):
    gait_type: GaitType = Field(GaitType.TRIPOD)
    step_height: float = Field(40, ge=20, le=80, description="Step height in mm")
    step_length: float = Field(35, ge=10, le=50, description="Step length in mm")
    speed: float = Field(1.0, ge=0.1, le=2.0, description="Speed multiplier")
    cycles: int = Field(1, ge=1, le=10, description="Number of gait cycles")

class MovementCommand(BaseModel):
    direction: Direction
    gait: GaitParameters = Field(default_factory=GaitParameters)
    duration: Optional[float] = Field(None, description="Duration in seconds (None for continuous)")

class SequenceStep(BaseModel):
    action: str = Field(..., description="Action type: move, pose, leg, servo, wait, led, buzzer")
    parameters: Dict = Field(..., description="Parameters for the action")
    duration: float = Field(1.0, ge=0.1, le=10.0)

class CalibrationData(BaseModel):
    leg_id: int = Field(..., ge=1, le=6)
    x_offset: int = Field(0, ge=-50, le=50)
    y_offset: int = Field(0, ge=-50, le=50)
    z_offset: int = Field(0, ge=-50, le=50)

class LedCommand(BaseModel):
    mode: LedMode
    color: Optional[List[int]] = Field(None, min_items=3, max_items=3, description="RGB values 0-255")
    brightness: Optional[int] = Field(None, ge=0, le=255)
    speed: Optional[float] = Field(1.0, ge=0.1, le=10.0)

class CameraCommand(BaseModel):
    mode: CameraMode
    filename: Optional[str] = None
    duration: Optional[int] = Field(None, ge=1, le=60, description="Video duration in seconds")

class SensorConfig(BaseModel):
    ultrasonic_enabled: bool = True
    imu_enabled: bool = True
    update_rate: float = Field(10.0, ge=1.0, le=100.0, description="Updates per second")

class EmergencyStop(BaseModel):
    stop_all: bool = True
    reason: str = Field("Emergency stop triggered")

# --- Utility Functions ---

def interpolate_servo(start: int, end: int, duration: float, steps: int = 50):
    """Generate interpolated servo positions for smooth movement."""
    positions = []
    for i in range(steps + 1):
        t = i / steps
        # Use sine wave for smooth acceleration/deceleration
        t_smooth = 0.5 * (1 - math.cos(math.pi * t))
        position = start + (end - start) * t_smooth
        positions.append(int(position))
    return positions, duration / steps

async def smooth_servo_move(channel: int, target: int, duration: float):
    """Move servo smoothly to target position."""
    current = servo_state.get(channel, 90)
    if current == target:
        return
    
    positions, delay = interpolate_servo(current, target, duration)
    for pos in positions:
        servo_controller.set_servo_angle(channel, pos)
        servo_state[channel] = pos
        await asyncio.sleep(delay)

async def move_leg_smooth(leg_id: int, angles: List[int], duration: float):
    """Move all servos in a leg smoothly and simultaneously."""
    channels = LEG_MAP[leg_id]
    tasks = []
    for channel, angle in zip(channels, angles):
        task = smooth_servo_move(channel, angle, duration)
        tasks.append(task)
    await asyncio.gather(*tasks)

def calculate_leg_angles(leg_id: int, x: float, y: float, z: float) -> List[int]:
    """Calculate servo angles for a leg position using inverse kinematics."""
    # Use the control system's inverse kinematics
    control.leg_positions[leg_id - 1] = [x, y, z]
    control.set_leg_angles()
    
    # Extract the calculated angles
    return [
        servo_state[LEG_MAP[leg_id][0]],
        servo_state[LEG_MAP[leg_id][1]],
        servo_state[LEG_MAP[leg_id][2]]
    ]

# --- High-Level Movement Functions ---

async def execute_tripod_gait(direction: Direction, params: GaitParameters, duration: Optional[float]):
    """Execute tripod gait movement pattern."""
    global is_moving
    is_moving = True
    
    direction_map = {
        Direction.FORWARD: (0, params.step_length, 0),
        Direction.BACKWARD: (0, -params.step_length, 0),
        Direction.LEFT: (-params.step_length, 0, 0),
        Direction.RIGHT: (params.step_length, 0, 0),
        Direction.TURN_LEFT: (0, 0, -10),
        Direction.TURN_RIGHT: (0, 0, 10),
        Direction.STOP: (0, 0, 0)
    }
    
    x_move, y_move, rotation = direction_map[direction]
    
    gait_data = ['CMD_MOVE', '1', str(int(x_move)), str(int(y_move)), 
                 str(int(10 / params.speed)), str(int(rotation))]
    
    start_time = time.time()
    while is_moving:
        control.run_gait(gait_data)
        
        if duration and (time.time() - start_time) >= duration:
            break
        
        if direction == Direction.STOP:
            break
            
        await asyncio.sleep(0.01)
    
    is_moving = False

async def execute_wave_gait(direction: Direction, params: GaitParameters, duration: Optional[float]):
    """Execute wave gait movement pattern."""
    global is_moving
    is_moving = True
    
    direction_map = {
        Direction.FORWARD: (0, params.step_length, 0),
        Direction.BACKWARD: (0, -params.step_length, 0),
        Direction.LEFT: (-params.step_length, 0, 0),
        Direction.RIGHT: (params.step_length, 0, 0),
        Direction.TURN_LEFT: (0, 0, -10),
        Direction.TURN_RIGHT: (0, 0, 10),
        Direction.STOP: (0, 0, 0)
    }
    
    x_move, y_move, rotation = direction_map[direction]
    
    gait_data = ['CMD_MOVE', '2', str(int(x_move)), str(int(y_move)), 
                 str(int(10 / params.speed)), str(int(rotation))]
    
    start_time = time.time()
    while is_moving:
        control.run_gait(gait_data)
        
        if duration and (time.time() - start_time) >= duration:
            break
        
        if direction == Direction.STOP:
            break
            
        await asyncio.sleep(0.01)
    
    is_moving = False

async def adjust_body_pose(pose: BodyPose):
    """Adjust robot body pose smoothly."""
    control.body_height = pose.height
    
    points = control.calculate_posture_balance(pose.roll, pose.pitch, pose.yaw)
    
    for i in range(6):
        points[i][0] += pose.x_offset
        points[i][1] += pose.y_offset
    
    control.transform_coordinates(points)
    
    steps = int(pose.duration * 50)
    for _ in range(steps):
        control.set_leg_angles()
        await asyncio.sleep(pose.duration / steps)

# --- API Endpoints ---

@app.get("/")
def root():
    """API root endpoint with comprehensive info."""
    return {
        "title": "Hexapod Robot Control API",
        "version": "3.0.0",
        "endpoints": {
            "movement": ["/move", "/stop", "/body/pose", "/gaits"],
            "servos": ["/servo/{channel}", "/servo/multi", "/servo/all"],
            "legs": ["/leg/{leg_id}/position", "/leg/{leg_id}/angles"],
            "sensors": ["/sensors", "/imu", "/distance", "/battery"],
            "head": ["/head", "/head/center", "/head/look/{direction}"],
            "control": ["/calibrate", "/relax", "/emergency-stop"],
            "peripherals": ["/led", "/buzzer", "/camera"],
            "advanced": ["/sequence", "/telemetry", "/config"]
        }
    }

@app.get("/status")
def get_status():
    """Get comprehensive robot status."""
    battery = adc.read_battery_voltage()
    imu_data = None
    try:
        roll, pitch, yaw = imu.update_imu_state()
        imu_data = {"roll": roll, "pitch": pitch, "yaw": yaw}
    except:
        pass
    
    return {
        "is_moving": is_moving,
        "servo_state": servo_state,
        "head_position": head_state,
        "battery": {
            "voltage1": battery[0],
            "voltage2": battery[1],
            "percentage": max(0, min(100, int((battery[0] - 5.0) / 3.4 * 100)))
        },
        "body_height": control.body_height,
        "distance": ultrasonic.get_distance(),
        "imu": imu_data,
        "timestamp": datetime.now().isoformat()
    }

# --- Servo Control Endpoints ---

@app.post("/servo/{channel}")
async def move_servo(channel: int, position: ServoPosition):
    """Move a single servo to target position."""
    if channel not in range(32):
        raise HTTPException(status_code=400, detail="Invalid channel")
    
    await smooth_servo_move(channel, position.angle, position.duration)
    return {"channel": channel, "angle": position.angle}

@app.post("/servo/multi")
async def move_multiple_servos(cmd: MultiServoPosition):
    """Move multiple servos with optional synchronization."""
    tasks = []
    for pos in cmd.positions:
        if pos.channel not in range(32):
            raise HTTPException(status_code=400, detail=f"Invalid channel: {pos.channel}")
        
        if cmd.synchronized:
            tasks.append(smooth_servo_move(pos.channel, pos.angle, pos.duration))
        else:
            await smooth_servo_move(pos.channel, pos.angle, pos.duration)
    
    if cmd.synchronized and tasks:
        await asyncio.gather(*tasks)
    
    return {"status": "completed", "count": len(cmd.positions)}

@app.post("/servo/all")
async def set_all_servos(angle: int = 90, duration: float = 1.0):
    """Set all servos to the same angle."""
    tasks = []
    for channel in range(32):
        tasks.append(smooth_servo_move(channel, angle, duration))
    
    await asyncio.gather(*tasks)
    return {"status": "all servos set", "angle": angle}

# --- Leg Control Endpoints ---

@app.post("/leg/{leg_id}/position")
async def set_leg_position(leg_id: int, position: LegPosition):
    """Set leg position using inverse kinematics."""
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg ID")
    
    angles = calculate_leg_angles(leg_id, position.x, position.y, position.z)
    await move_leg_smooth(leg_id, angles, position.duration)
    
    return {"leg": leg_id, "position": position.dict(), "angles": angles}

@app.post("/leg/{leg_id}/angles")
async def set_leg_angles(leg_id: int, angles: LegAngles):
    """Set leg joint angles directly."""
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg ID")
    
    angle_list = [angles.coxa, angles.femur, angles.tibia]
    await move_leg_smooth(leg_id, angle_list, angles.duration)
    
    return {"leg": leg_id, "angles": angle_list}

@app.get("/leg/{leg_id}")
def get_leg_info(leg_id: int):
    """Get current leg position and angles."""
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg ID")
    
    channels = LEG_MAP[leg_id]
    angles = [servo_state[ch] for ch in channels]
    
    # Get position from control system
    position = control.leg_positions[leg_id - 1]
    
    return {
        "leg": leg_id,
        "channels": channels,
        "angles": {"coxa": angles[0], "femur": angles[1], "tibia": angles[2]},
        "position": {"x": position[0], "y": position[1], "z": position[2]}
    }

# --- Movement Control Endpoints ---

@app.post("/body/pose")
async def set_body_pose(pose: BodyPose):
    """Adjust robot body pose."""
    await adjust_body_pose(pose)
    return {"pose": pose.dict(), "status": "adjusted"}

@app.post("/move")
async def move_robot(command: MovementCommand, background_tasks: BackgroundTasks):
    """Execute movement with specified gait and direction."""
    global movement_task, is_moving
    
    if is_moving:
        is_moving = False
        if movement_task:
            movement_task.cancel()
    
    if command.direction == Direction.STOP:
        return {"status": "stopped"}
    
    if command.gait.gait_type == GaitType.TRIPOD:
        gait_func = execute_tripod_gait
    elif command.gait.gait_type == GaitType.WAVE:
        gait_func = execute_wave_gait
    else:
        raise HTTPException(status_code=400, detail="Gait type not implemented")
    
    background_tasks.add_task(
        gait_func, 
        command.direction, 
        command.gait, 
        command.duration
    )
    
    return {"status": "moving", "command": command.dict()}

@app.post("/stop")
def stop_movement():
    """Stop all movement immediately."""
    global is_moving
    is_moving = False
    control.command_queue = ['', '', '', '', '', '']
    return {"status": "stopped"}

@app.post("/emergency-stop")
def emergency_stop(cmd: EmergencyStop):
    """Emergency stop - halt all operations."""
    global is_moving
    
    # Stop movement
    is_moving = False
    control.command_queue = ['', '', '', '', '', '']
    
    # Relax servos if requested
    if cmd.stop_all:
        control.relax(True)
        
    # Turn off peripherals
    buzzer.set_state(False)
    if _led:
        _led.color_wipe([0, 0, 0])
    
    return {
        "status": "emergency stop activated",
        "reason": cmd.reason,
        "timestamp": datetime.now().isoformat()
    }

# --- Calibration Endpoints ---

@app.post("/calibrate")
def calibrate_robot():
    """Calibrate robot to default position."""
    control.calibrate()
    control.set_leg_angles()
    return {"status": "calibrated"}

@app.post("/calibrate/leg")
def calibrate_leg(data: CalibrationData):
    """Calibrate individual leg offsets."""
    leg_idx = data.leg_id - 1
    control.calibration_leg_positions[leg_idx][0] = data.x_offset
    control.calibration_leg_positions[leg_idx][1] = data.y_offset
    control.calibration_leg_positions[leg_idx][2] = data.z_offset
    control.calibrate()
    control.set_leg_angles()
    
    return {"leg": data.leg_id, "offsets": [data.x_offset, data.y_offset, data.z_offset]}

@app.post("/calibrate/save")
def save_calibration():
    """Save current calibration data."""
    control.save_to_txt(control.calibration_leg_positions, 'point')
    return {"status": "calibration saved"}

# --- Sensor Endpoints ---

@app.get("/sensors")
def get_all_sensors():
    """Get all sensor readings."""
    battery = adc.read_battery_voltage()
    
    imu_data = None
    try:
        roll, pitch, yaw = imu.update_imu_state()
        imu_data = {"roll": roll, "pitch": pitch, "yaw": yaw}
    except:
        pass
    
    return {
        "ultrasonic": {
            "distance_cm": ultrasonic.get_distance(),
            "unit": "centimeters"
        },
        "battery": {
            "voltage1": battery[0],
            "voltage2": battery[1],
            "percentage": max(0, min(100, int((battery[0] - 5.0) / 3.4 * 100))),
            "unit": "volts"
        },
        "imu": imu_data
    }

@app.get("/imu")
def get_imu_data():
    """Get detailed IMU data."""
    try:
        roll, pitch, yaw = imu.update_imu_state()
        accel_data = imu.sensor.get_accel_data()
        gyro_data = imu.sensor.get_gyro_data()
        
        return {
            "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
            "accelerometer": accel_data,
            "gyroscope": gyro_data,
            "quaternion": {
                "w": imu.quaternion_w,
                "x": imu.quaternion_x,
                "y": imu.quaternion_y,
                "z": imu.quaternion_z
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IMU error: {str(e)}")

@app.post("/imu/balance")
def enable_imu_balance(enable: bool = True):
    """Enable or disable IMU-based self-balancing."""
    if enable:
        control.command_queue = ["CMD_BALANCE", "1"]
    else:
        control.command_queue = ["CMD_BALANCE", "0"]
    return {"imu_balance": enable}

# --- LED Control Endpoints ---

@app.post("/led")
def control_led(cmd: LedCommand):
    """Control LED patterns and colors."""
    global _led
    if _led is None:
        _led = Led()
    
    if cmd.mode == LedMode.OFF:
        _led.color_wipe([0, 0, 0])
    elif cmd.mode == LedMode.SOLID and cmd.color:
        _led.led_index(0x7f, cmd.color[0], cmd.color[1], cmd.color[2])
    elif cmd.mode == LedMode.RAINBOW:
        _led.rainbow(wait_ms=int(20 / (cmd.speed or 1)))
    elif cmd.mode == LedMode.THEATER and cmd.color:
        _led.theater_chase(cmd.color, wait_ms=int(50 / (cmd.speed or 1)))
    
    if cmd.brightness is not None and _led:
        _led.strip.set_led_brightness(cmd.brightness)
    
    return {"status": "led command executed", "mode": cmd.mode}

# --- Camera Endpoints ---

@app.post("/camera")
def control_camera(cmd: CameraCommand):
    """Control camera operations."""
    if cmd.mode == CameraMode.OFF:
        camera.stop_stream()
        return {"status": "camera stopped"}
    
    elif cmd.mode == CameraMode.PREVIEW:
        camera.start_image()
        return {"status": "preview started"}
    
    elif cmd.mode == CameraMode.CAPTURE:
        filename = cmd.filename or f"capture_{int(time.time())}.jpg"
        metadata = camera.save_image(filename)
        return {"status": "image captured", "filename": filename, "metadata": metadata}
    
    elif cmd.mode == CameraMode.STREAM:
        if cmd.filename:
            camera.save_video(cmd.filename, cmd.duration or 10)
            return {"status": "video saved", "filename": cmd.filename}
        else:
            camera.start_stream()
            return {"status": "streaming started"}

@app.get("/camera/stream")
async def get_camera_stream():
    """Get live camera stream."""
    def generate():
        camera.start_stream()
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
@app.post("/head")
async def move_head(position: HeadPosition):
    """Control head pan and tilt movements."""
    global head_state
    tasks = []
    
    if position.pan is not None:
        # Channel 1 controls pan (left/right)
        # Original code uses 180 - angle for pan
        pan_angle = 180 - position.pan
        tasks.append(smooth_servo_move(1, pan_angle, position.duration))
        head_state["pan"] = position.pan
    
    if position.tilt is not None:
        # Channel 0 controls tilt (up/down)
        tasks.append(smooth_servo_move(0, position.tilt, position.duration))
        head_state["tilt"] = position.tilt
    
    if tasks:
        await asyncio.gather(*tasks)
    
    return {"head": head_state, "status": "positioned"}

@app.get("/head")
def get_head_position():
    """Get current head position."""
    return {
        "pan": head_state["pan"],
        "tilt": head_state["tilt"],
        "servo_angles": {
            "pan_servo": servo_state[1],
            "tilt_servo": servo_state[0]
        }
    }

@app.post("/head/center")
async def center_head(duration: float = 1.0):
    """Center the head to default position."""
    await move_head(HeadPosition(pan=90, tilt=90, duration=duration))
    return {"status": "head centered"}

@app.post("/head/look/{direction}")
async def look_direction(direction: str, duration: float = 1.0):
    """Make head look in a specific direction."""
    positions = {
        "up": HeadPosition(tilt=50, duration=duration),
        "down": HeadPosition(tilt=140, duration=duration),
        "left": HeadPosition(pan=140, duration=duration),
        "right": HeadPosition(pan=40, duration=duration),
        "forward": HeadPosition(pan=90, tilt=90, duration=duration)
    }
    
    if direction not in positions:
        raise HTTPException(status_code=400, detail="Invalid direction")
    
    await move_head(positions[direction])
    return {"status": f"looking {direction}"}

@app.post("/head/track")
async def enable_head_tracking(enable: bool = True):
    """Enable or disable head tracking (requires additional implementation)."""
    # This would integrate with camera/vision system for object tracking
    return {"tracking": enable, "note": "Tracking logic requires vision system integration"}

# --- Advanced Features ---

@app.post("/sequence")
async def execute_sequence(steps: List[SequenceStep]):
    """Execute a sequence of movements and actions."""
    results = []
    
    for i, step in enumerate(steps):
        try:
            if step.action == "move":
                cmd = MovementCommand(**step.parameters)
                await move_robot(cmd, BackgroundTasks())
                await asyncio.sleep(step.duration)
                
            elif step.action == "pose":
                pose = BodyPose(**step.parameters)
                await set_body_pose(pose)
                
            elif step.action == "leg":
                leg_id = step.parameters.get("leg_id")
                if "position" in step.parameters:
                    position = LegPosition(**step.parameters.get("position"))
                    await set_leg_position(leg_id, position)
                elif "angles" in step.parameters:
                    angles = LegAngles(**step.parameters.get("angles"))
                    await set_leg_angles(leg_id, angles)
                    
            elif step.action == "servo":
                pos = ServoPosition(**step.parameters)
                await move_servo(pos.channel, pos)
            
            elif step.action == "head":
                head_pos = HeadPosition(**step.parameters)
                await move_head(head_pos)
                
            elif step.action == "led":
                led_cmd = LedCommand(**step.parameters)
                control_led(led_cmd)
                
            elif step.action == "buzzer":
                buzzer.set_state(step.parameters.get("state", False))
                
            elif step.action == "wait":
                await asyncio.sleep(step.duration)
            
            results.append({"step": i, "action": step.action, "status": "completed"})
            
        except Exception as e:
            results.append({"step": i, "action": step.action, "status": "failed", "error": str(e)})
            break
    
    return {"sequence": results, "total_steps": len(steps)}

@app.websocket("/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry data."""
    await websocket.accept()
    telemetry_clients.append(websocket)
    
    try:
        while True:
            # Send telemetry data every 100ms
            status = get_status()
            await websocket.send_json(status)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        telemetry_clients.remove(websocket)

# --- Configuration Management ---

@app.get("/config")
def get_configuration():
    """Get current robot configuration."""
    return {
        "leg_map": LEG_MAP,
        "calibration": control.calibration_leg_positions,
        "body_height": control.body_height,
        "servo_state": servo_state,
        "config": config
    }

@app.post("/config")
def update_configuration(new_config: Dict[str, Any]):
    """Update robot configuration."""
    global config
    config.update(new_config)
    return {"status": "configuration updated", "config": config}

# --- Demo and Preset Endpoints ---

@app.get("/demos")
def list_demos():
    """List available demonstration routines."""
    return {
        "demos": [
            {"name": "wave", "description": "Wave hello with front leg"},
            {"name": "dance", "description": "Simple dance routine"},
            {"name": "walk_square", "description": "Walk in a square pattern"},
            {"name": "pushups", "description": "Do pushup exercises"},
            {"name": "sit", "description": "Sit down position"},
            {"name": "stand_tall", "description": "Stand at maximum height"},
            {"name": "look_around", "description": "Look around with head movements"},
            {"name": "nod", "description": "Nod head up and down"}
        ]
    }

@app.post("/demo/{demo_name}")
async def run_demo(demo_name: str):
    """Run pre-programmed demonstration sequences."""
    if demo_name == "wave":
        await set_leg_position(1, LegPosition(x=140, y=50, z=50, duration=1.0))
        await asyncio.sleep(0.5)
        await set_leg_position(1, LegPosition(x=140, y=-50, z=50, duration=0.5))
        await set_leg_position(1, LegPosition(x=140, y=50, z=50, duration=0.5))
        await set_leg_position(1, LegPosition(x=140, y=0, z=0, duration=1.0))
        
    elif demo_name == "dance":
        poses = [
            BodyPose(height=-10, roll=10, duration=0.5),
            BodyPose(height=-10, roll=-10, duration=0.5),
            BodyPose(height=-40, pitch=10, duration=0.5),
            BodyPose(height=-40, pitch=-10, duration=0.5),
            BodyPose(height=-25, duration=1.0)
        ]
        for pose in poses:
            await set_body_pose(pose)
            
    elif demo_name == "walk_square":
        movements = [
            MovementCommand(direction=Direction.FORWARD, duration=3),
            MovementCommand(direction=Direction.RIGHT, duration=3),
            MovementCommand(direction=Direction.BACKWARD, duration=3),
            MovementCommand(direction=Direction.LEFT, duration=3),
        ]
        for move in movements:
            await move_robot(move, BackgroundTasks())
            await asyncio.sleep(move.duration + 0.5)
            
    elif demo_name == "pushups":
        for _ in range(5):
            await set_body_pose(BodyPose(height=-10, duration=1.0))
            await set_body_pose(BodyPose(height=-40, duration=1.0))
            
    elif demo_name == "sit":
        await set_body_pose(BodyPose(height=-50, pitch=-15, duration=2.0))
        
    elif demo_name == "stand_tall":
        await set_body_pose(BodyPose(height=-50, duration=2.0))
    
    elif demo_name == "look_around":
        # Look around demonstration
        await move_head(HeadPosition(pan=90, tilt=90, duration=1.0))
        await move_head(HeadPosition(pan=40, duration=1.5))
        await move_head(HeadPosition(pan=140, duration=2.0))
        await move_head(HeadPosition(pan=90, duration=1.5))
        await move_head(HeadPosition(tilt=50, duration=1.0))
        await move_head(HeadPosition(tilt=140, duration=1.5))
        await move_head(HeadPosition(pan=90, tilt=90, duration=1.0))
        
    elif demo_name == "nod":
        # Nod head up and down
        for _ in range(3):
            await move_head(HeadPosition(tilt=70, duration=0.5))
            await move_head(HeadPosition(tilt=110, duration=0.5))
        await move_head(HeadPosition(tilt=90, duration=0.5))
        
    else:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    return {"demo": demo_name, "status": "completed"}

# --- System Control Endpoints ---

@app.post("/relax")
def relax_servos(state: bool = True):
    """Relax or unrelax all servos."""
    control.relax(state)
    return {"relaxed": state}

@app.post("/reboot")
def reboot_system():
    """Reboot the robot system."""
    # Add actual reboot logic here
    return {"status": "reboot scheduled", "delay": 5}

@app.get("/health")
def health_check():
    """System health check."""
    battery = adc.read_battery_voltage()
    
    return {
        "status": "healthy",
        "battery_ok": battery[0] > 6.0 and battery[1] > 7.0,
        "servos_ok": True,  # Add actual servo check
        "sensors_ok": True,  # Add actual sensor check
        "uptime": time.time()  # Add actual uptime
    }

# --- Startup and Shutdown ---

@app.on_event("startup")
async def startup_event():
    """Initialize robot on API startup."""
    print("Hexapod Robot API starting up...")
    control.condition_thread.start()
    
@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of robot systems."""
    print("Hexapod Robot API shutting down...")
    global is_moving
    is_moving = False
    control.relax(True)
    if _led:
        _led.color_wipe([0, 0, 0])
    buzzer.set_state(False)
    camera.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)