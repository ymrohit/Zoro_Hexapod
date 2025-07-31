"""Enhanced FastAPI server for natural hexapod robot control with smooth movements."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import time
import math
import numpy as np
from enum import Enum

from servo import Servo
from led import Led
from buzzer import Buzzer
from ultrasonic import Ultrasonic
from adc import ADC
from control import Control

app = FastAPI(title="Enhanced Hexapod Control API")

# Initialize controllers
servo_controller = Servo()
_led = None
buzzer = Buzzer()
ultrasonic = Ultrasonic()
adc = ADC()
control = Control()

# Global state
servo_state = {i: 90 for i in range(32)}
movement_task = None
is_moving = False

# Leg and servo mapping
LEG_MAP = {
    1: [15, 14, 13],  # Right front
    2: [12, 11, 10],  # Right middle
    3: [9, 8, 31],    # Right back
    4: [22, 23, 27],  # Left back
    5: [19, 20, 21],  # Left middle
    6: [16, 17, 18],  # Left front
}

# Gait types
class GaitType(str, Enum):
    TRIPOD = "tripod"
    WAVE = "wave"
    RIPPLE = "ripple"
    CUSTOM = "custom"

# Movement directions
class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"

# --- Pydantic Models ---

class ServoPosition(BaseModel):
    channel: int = Field(..., ge=0, le=31)
    angle: int = Field(..., ge=0, le=180)
    duration: float = Field(1.0, ge=0.1, le=10.0, description="Duration in seconds")

class LegPosition(BaseModel):
    x: float = Field(..., description="X coordinate in mm")
    y: float = Field(..., description="Y coordinate in mm")
    z: float = Field(..., description="Z coordinate in mm")
    duration: float = Field(1.0, ge=0.1, le=10.0)

class BodyPose(BaseModel):
    height: float = Field(default=-40, ge=-50, le=50, description="Body height adjustment")
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
    action: str = Field(..., description="Action type: move, pose, leg, wait")
    parameters: Dict = Field(..., description="Parameters for the action")
    duration: float = Field(1.0, ge=0.1, le=10.0)

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
    # Transform coordinates based on leg position
    angle_offset = (leg_id - 1) * 60  # Degrees between legs
    
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
    
    # Map direction to x, y, and rotation
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
    
    # Use existing control system gait
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
    
    # Similar to tripod but using gait mode 2
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
    # Set body height
    control.body_height = pose.height
    
    # Calculate posture balance
    points = control.calculate_posture_balance(pose.roll, pose.pitch, pose.yaw)
    
    # Apply position offsets
    for i in range(6):
        points[i][0] += pose.x_offset
        points[i][1] += pose.y_offset
    
    # Transform and apply
    control.transform_coordinates(points)
    
    # Smooth transition
    steps = int(pose.duration * 50)  # 50 updates per second
    for _ in range(steps):
        control.set_leg_angles()
        await asyncio.sleep(pose.duration / steps)

# --- API Endpoints ---

@app.get("/")
def root():
    """API root endpoint with basic info."""
    return {
        "title": "Enhanced Hexapod Control API",
        "version": "2.0",
        "features": [
            "Smooth servo movements",
            "Natural gait patterns",
            "Body pose control",
            "Movement sequences",
            "Real-time sensor data"
        ]
    }

@app.get("/status")
def get_status():
    """Get current robot status."""
    battery = adc.read_battery_voltage()
    return {
        "is_moving": is_moving,
        "servo_state": servo_state,
        "battery": {"v1": battery[0], "v2": battery[1]},
        "body_height": control.body_height,
        "distance": ultrasonic.get_distance()
    }

@app.post("/servo/smooth")
async def move_servo_smooth(position: ServoPosition):
    """Move a servo smoothly to target position."""
    await smooth_servo_move(position.channel, position.angle, position.duration)
    return {"channel": position.channel, "angle": position.angle, "smooth": True}

@app.post("/leg/{leg_id}/position")
async def set_leg_position(leg_id: int, position: LegPosition):
    """Set leg position using inverse kinematics with smooth movement."""
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg ID")
    
    angles = calculate_leg_angles(leg_id, position.x, position.y, position.z)
    await move_leg_smooth(leg_id, angles, position.duration)
    
    return {"leg": leg_id, "position": position.dict(), "angles": angles}

@app.post("/body/pose")
async def set_body_pose(pose: BodyPose):
    """Adjust robot body pose (height, roll, pitch, yaw)."""
    await adjust_body_pose(pose)
    return {"pose": pose.dict(), "status": "adjusted"}

@app.post("/move")
async def move_robot(command: MovementCommand, background_tasks: BackgroundTasks):
    """Execute movement with specified gait and direction."""
    global movement_task, is_moving
    
    # Cancel existing movement
    if is_moving:
        is_moving = False
        if movement_task:
            movement_task.cancel()
    
    if command.direction == Direction.STOP:
        return {"status": "stopped"}
    
    # Select gait function
    if command.gait.gait_type == GaitType.TRIPOD:
        gait_func = execute_tripod_gait
    elif command.gait.gait_type == GaitType.WAVE:
        gait_func = execute_wave_gait
    else:
        raise HTTPException(status_code=400, detail="Gait type not implemented")
    
    # Execute movement in background
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

@app.post("/calibrate")
def calibrate_robot():
    """Calibrate robot to default position."""
    control.calibrate()
    control.set_leg_angles()
    return {"status": "calibrated"}

@app.post("/sequence")
async def execute_sequence(steps: List[SequenceStep]):
    """Execute a sequence of movements and actions."""
    results = []
    
    for step in steps:
        if step.action == "move":
            cmd = MovementCommand(**step.parameters)
            await move_robot(cmd, BackgroundTasks())
            await asyncio.sleep(step.duration)
            
        elif step.action == "pose":
            pose = BodyPose(**step.parameters)
            await set_body_pose(pose)
            
        elif step.action == "leg":
            leg_id = step.parameters.get("leg_id")
            position = LegPosition(**step.parameters.get("position"))
            await set_leg_position(leg_id, position)
            
        elif step.action == "wait":
            await asyncio.sleep(step.duration)
        
        results.append({"step": step.dict(), "completed": True})
    
    return {"sequence": results, "status": "completed"}

@app.get("/gaits")
def list_gaits():
    """List available gait patterns."""
    return {
        "gaits": [
            {
                "name": "tripod",
                "description": "Fast tripod gait - 3 legs move at once",
                "stability": "medium",
                "speed": "fast"
            },
            {
                "name": "wave",
                "description": "Slow wave gait - 1 leg moves at a time",
                "stability": "high",
                "speed": "slow"
            },
            {
                "name": "ripple",
                "description": "Ripple gait - 2 legs move at once",
                "stability": "high",
                "speed": "medium"
            }
        ]
    }

@app.post("/demo/{demo_name}")
async def run_demo(demo_name: str):
    """Run pre-programmed demonstration sequences."""
    if demo_name == "wave":
        # Wave hello with front leg
        await set_leg_position(1, LegPosition(x=140, y=50, z=50, duration=1.0))
        await asyncio.sleep(0.5)
        await set_leg_position(1, LegPosition(x=140, y=-50, z=50, duration=0.5))
        await set_leg_position(1, LegPosition(x=140, y=50, z=50, duration=0.5))
        await set_leg_position(1, LegPosition(x=140, y=0, z=0, duration=1.0))
        
    elif demo_name == "dance":
        # Simple dance routine
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
        # Walk in a square pattern
        movements = [
            MovementCommand(direction=Direction.FORWARD, duration=3),
            MovementCommand(direction=Direction.RIGHT, duration=3),
            MovementCommand(direction=Direction.BACKWARD, duration=3),
            MovementCommand(direction=Direction.LEFT, duration=3),
        ]
        for move in movements:
            await move_robot(move, BackgroundTasks())
            await asyncio.sleep(move.duration + 0.5)
            
    else:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    return {"demo": demo_name, "status": "completed"}

@app.post("/relax")
def relax_servos(state: bool = True):
    """Relax or unrelax all servos."""
    control.relax(state)
    return {"relaxed": state}

@app.get("/sensors")
def get_sensor_data():
    """Get all sensor readings."""
    battery = adc.read_battery_voltage()
    return {
        "ultrasonic": {
            "distance_cm": ultrasonic.get_distance(),
            "unit": "centimeters"
        },
        "battery": {
            "voltage1": battery[0],
            "voltage2": battery[1],
            "unit": "volts"
        },
        "imu": {
            "available": hasattr(control, 'imu'),
            "note": "Use /imu endpoint for IMU data"
        }
    }

@app.post("/imu/balance")
def enable_imu_balance(enable: bool = True):
    """Enable or disable IMU-based self-balancing."""
    if enable:
        control.command_queue = [cmd.CMD_BALANCE, '1']
    else:
        control.command_queue = [cmd.CMD_BALANCE, '0']
    return {"imu_balance": enable}

if __name__ == "__main__":
    import uvicorn
    # Start the control thread
    control.condition_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)