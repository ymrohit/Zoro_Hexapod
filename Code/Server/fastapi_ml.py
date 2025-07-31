"""
Enhanced FastAPI server for hexapod robot with ML autonomous control
Combines smooth movement control with machine learning capabilities
"""

# Standard library imports
import asyncio
import time
import math
import random
import pickle
import json
from enum import Enum
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Robot hardware imports
from servo import Servo
from led import Led
from buzzer import Buzzer
from ultrasonic import Ultrasonic
from adc import ADC
from control import Control
from command import COMMAND as cmd

# Initialize FastAPI app
app = FastAPI(title="Enhanced Hexapod Control API with ML")

# Initialize hardware controllers
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
ml_controller = None

# Leg and servo mapping
LEG_MAP = {
    1: [15, 14, 13],  # Right front
    2: [12, 11, 10],  # Right middle
    3: [9, 8, 31],    # Right back
    4: [22, 23, 27],  # Left back
    5: [19, 20, 21],  # Left middle
    6: [16, 17, 18],  # Left front
}

# ========== ENUMS ==========

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

class ActionSpace(Enum):
    """Possible robot actions for ML"""
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    MOVE_LEFT = 4
    MOVE_RIGHT = 5
    INCREASE_HEIGHT = 6
    DECREASE_HEIGHT = 7
    STOP = 8
    ADJUST_TILT = 9

class TerrainType(Enum):
    """Detected terrain types"""
    FLAT = "flat"
    OBSTACLE = "obstacle"
    SLOPE = "slope"
    ROUGH = "rough"
    STAIRS = "stairs"

# ========== PYDANTIC MODELS ==========

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
    height: float = Field(default=-40, ge=-50, le=50, description="Body height adjustment")  # Changed from -25 to -40
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

# ========== ML DATA CLASSES ==========

@dataclass
class SensorData:
    """Sensor readings at a specific time"""
    ultrasonic_distance: float
    battery_voltage: Tuple[float, float]
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0

@dataclass
class RobotState:
    """Complete robot state"""
    sensor_data: SensorData
    body_height: float
    body_pose: Dict[str, float]
    is_moving: bool
    current_action: Optional[str] = None

# ========== NEURAL NETWORKS ==========

class HexapodBrain(nn.Module):
    """Neural network for autonomous decision making"""
    
    def __init__(self, input_size=10, hidden_size=128, output_size=len(ActionSpace)):
        super(HexapodBrain, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)
        return x

class TerrainClassifier(nn.Module):
    """Classify terrain based on sensor data"""
    
    def __init__(self, input_size=7, hidden_size=64, output_size=len(TerrainType)):
        super(TerrainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# ========== ML COMPONENTS ==========

class ExperienceBuffer:
    """Store experiences for training"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ========== UTILITY FUNCTIONS ==========

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

# ========== MOVEMENT FUNCTIONS ==========

async def execute_tripod_gait(direction: Direction, params: GaitParameters, duration: Optional[float]):
    """Execute tripod gait movement pattern."""
    global is_moving
    is_moving = True
    
    # FIXED: Swapped X and Y coordinates for proper forward movement
    direction_map = {
        Direction.FORWARD: (params.step_length, 0, 0),      # Changed from (0, params.step_length, 0)
        Direction.BACKWARD: (-params.step_length, 0, 0),    # Changed from (0, -params.step_length, 0)
        Direction.LEFT: (0, params.step_length, 0),         # Changed from (-params.step_length, 0, 0)
        Direction.RIGHT: (0, -params.step_length, 0),       # Changed from (params.step_length, 0, 0)
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

# Also fix execute_wave_gait the same way:
async def execute_wave_gait(direction: Direction, params: GaitParameters, duration: Optional[float]):
    """Execute wave gait movement pattern."""
    global is_moving
    is_moving = True
    
    # FIXED: Swapped X and Y coordinates
    direction_map = {
        Direction.FORWARD: (params.step_length, 0, 0),      # Changed
        Direction.BACKWARD: (-params.step_length, 0, 0),    # Changed
        Direction.LEFT: (0, params.step_length, 0),         # Changed
        Direction.RIGHT: (0, -params.step_length, 0),       # Changed
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

# ========== AUTONOMOUS CONTROLLER ==========

class AutonomousController:
    """Main ML controller for autonomous movement"""
    
    def __init__(self, robot_control, servo_controller, sensors):
        self.control = robot_control
        self.servo = servo_controller
        self.ultrasonic = sensors['ultrasonic']
        self.adc = sensors['adc']
        self.imu = sensors.get('imu', None)
        
        # ML Models
        self.brain = HexapodBrain()
        self.terrain_classifier = TerrainClassifier()
        self.experience_buffer = ExperienceBuffer()
        
        # Learning parameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.optimizer = optim.Adam(self.brain.parameters(), lr=self.learning_rate)
        
        # State tracking
        self.previous_state = None
        self.previous_action = None
        self.step_count = 0
        self.episode_rewards = []
        
        # Safety parameters
        self.min_distance = 10.0  # cm
        self.max_tilt = 15.0  # degrees
        self.low_battery_threshold = 6.0  # volts
        
        # Load pre-trained models if available
        self.load_models()
    
    async def get_sensor_state(self) -> SensorData:
        """Collect current sensor readings"""
        distance = self.ultrasonic.get_distance()
        battery = self.adc.read_battery_voltage()
        
        # Get IMU data if available
        pitch, roll, yaw = 0, 0, 0
        if self.imu:
            try:
                pitch, roll, yaw = self.imu.update_imu_state()
            except:
                pass
        
        return SensorData(
            ultrasonic_distance=distance,
            battery_voltage=battery,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            timestamp=time.time()
        )
    
    def state_to_tensor(self, sensor_data: SensorData, body_height: float) -> torch.Tensor:
        """Convert sensor data to neural network input"""
        state = torch.tensor([
            sensor_data.ultrasonic_distance / 100.0,  # Normalize to 0-1
            sensor_data.battery_voltage[0] / 10.0,
            sensor_data.battery_voltage[1] / 10.0,
            sensor_data.pitch / 180.0,
            sensor_data.roll / 180.0,
            sensor_data.yaw / 180.0,
            (body_height + 50) / 100.0,  # Normalize height
            self.step_count / 1000.0,  # Time factor
            1.0 if is_moving else 0.0,
            self.epsilon  # Exploration factor
        ], dtype=torch.float32)
        return state
    
    def classify_terrain(self, sensor_data: SensorData) -> TerrainType:
        """Classify current terrain type"""
        features = torch.tensor([
            sensor_data.ultrasonic_distance / 100.0,
            sensor_data.pitch / 180.0,
            sensor_data.roll / 180.0,
            abs(sensor_data.pitch) / 180.0,
            abs(sensor_data.roll) / 180.0,
            1.0 if sensor_data.ultrasonic_distance < 20 else 0.0,
            1.0 if abs(sensor_data.pitch) > 5 or abs(sensor_data.roll) > 5 else 0.0
        ], dtype=torch.float32)
        
        with torch.no_grad():
            terrain_probs = self.terrain_classifier(features)
            terrain_idx = torch.argmax(terrain_probs).item()
            return list(TerrainType)[terrain_idx]
    
    def select_action(self, state_tensor: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, len(ActionSpace) - 1)
        else:
            # Exploitation: use neural network
            with torch.no_grad():
                action_probs = self.brain(state_tensor)
                return torch.argmax(action_probs).item()
    
    def calculate_reward(self, sensor_data: SensorData, action: int, 
                        previous_sensor_data: SensorData) -> float:
        """Calculate reward for reinforcement learning"""
        reward = 0.0
        
        # Distance-based rewards
        if sensor_data.ultrasonic_distance < self.min_distance:
            reward -= 10.0  # Penalty for being too close to obstacle
        elif sensor_data.ultrasonic_distance > 30:
            reward += 1.0  # Reward for clear path
        
        # Movement rewards
        if action == ActionSpace.MOVE_FORWARD.value:
            if sensor_data.ultrasonic_distance > 20:
                reward += 2.0  # Reward forward movement in clear areas
            else:
                reward -= 5.0  # Penalty for moving toward obstacle
        
        # Stability rewards
        tilt = abs(sensor_data.pitch) + abs(sensor_data.roll)
        if tilt < 5:
            reward += 1.0  # Reward for stability
        elif tilt > self.max_tilt:
            reward -= 5.0  # Penalty for excessive tilt
        
        # Battery conservation
        if sensor_data.battery_voltage[0] < self.low_battery_threshold:
            reward -= 0.5  # Small penalty for low battery
        
        # Exploration bonus
        if action != ActionSpace.STOP.value:
            reward += 0.1  # Small reward for any movement
        
        return reward
    
    async def execute_action(self, action: int) -> None:
        """Execute the selected action"""
        action_enum = ActionSpace(action)
        
        if action_enum == ActionSpace.MOVE_FORWARD:
            await self.move_robot("forward", duration=0.5)
        elif action_enum == ActionSpace.MOVE_BACKWARD:
            await self.move_robot("backward", duration=0.5)
        elif action_enum == ActionSpace.TURN_LEFT:
            await self.move_robot("turn_left", duration=0.3)
        elif action_enum == ActionSpace.TURN_RIGHT:
            await self.move_robot("turn_right", duration=0.3)
        elif action_enum == ActionSpace.MOVE_LEFT:
            await self.move_robot("left", duration=0.5)
        elif action_enum == ActionSpace.MOVE_RIGHT:
            await self.move_robot("right", duration=0.5)
        elif action_enum == ActionSpace.INCREASE_HEIGHT:
            await self.adjust_height(10)
        elif action_enum == ActionSpace.DECREASE_HEIGHT:
            await self.adjust_height(-10)
        elif action_enum == ActionSpace.STOP:
            global is_moving
            is_moving = False
        elif action_enum == ActionSpace.ADJUST_TILT:
            await self.auto_balance()
    
    async def move_robot(self, direction: str, duration: float = 0.5):
        """Execute movement command"""
        # FIXED: Swapped X and Y in the gait data
        gait_data = {
            "forward": ['CMD_MOVE', '1', '35', '0', '8', '0'],      # Changed from '0', '35'
            "backward": ['CMD_MOVE', '1', '-35', '0', '8', '0'],    # Changed from '0', '-35'
            "left": ['CMD_MOVE', '1', '0', '35', '8', '0'],         # Changed from '-35', '0'
            "right": ['CMD_MOVE', '1', '0', '-35', '8', '0'],       # Changed from '35', '0'
            "turn_left": ['CMD_MOVE', '1', '0', '0', '8', '-10'],
            "turn_right": ['CMD_MOVE', '1', '0', '0', '8', '10']
        }
        
        if direction in gait_data:
            self.control.run_gait(gait_data[direction])
            await asyncio.sleep(duration)

    
    async def adjust_height(self, delta: int):
        """Adjust robot height"""
        current_height = self.control.body_height
        new_height = max(-40, min(30, current_height + delta))
        
        # Smooth height adjustment
        points = self.control.calculate_posture_balance(0, 0, 0)
        for i in range(6):
            points[i][2] = new_height
        self.control.body_height = new_height
        self.control.transform_coordinates(points)
        self.control.set_leg_angles()
        await asyncio.sleep(0.5)
    
    async def auto_balance(self):
        """Automatic balancing using IMU"""
        if not self.imu:
            return
        
        pitch, roll, _ = self.imu.update_imu_state()
        
        # Simple proportional control for balance
        pitch_correction = -pitch * 0.5
        roll_correction = -roll * 0.5
        
        points = self.control.calculate_posture_balance(
            roll_correction, pitch_correction, 0
        )
        self.control.transform_coordinates(points)
        self.control.set_leg_angles()
        await asyncio.sleep(0.3)
    
    def train_step(self, state, action, reward, next_state, done):
        """Single training step for the neural network"""
        self.experience_buffer.push(state, action, reward, next_state, done)
        
        if len(self.experience_buffer) < 32:
            return
        
        # Sample batch from experience buffer
        batch = self.experience_buffer.sample(32)
        states = torch.stack([s for s, _, _, _, _ in batch])
        actions = torch.tensor([a for _, a, _, _, _ in batch])
        rewards = torch.tensor([r for _, _, r, _, _ in batch])
        next_states = torch.stack([ns for _, _, _, ns, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch])
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.brain(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())
        
        # Calculate current Q-values
        current_q_values = self.brain(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    async def autonomous_step(self) -> Dict:
        """Execute one step of autonomous movement"""
        # Get current sensor state
        sensor_data = await self.get_sensor_state()
        state_tensor = self.state_to_tensor(sensor_data, self.control.body_height)
        
        # Classify terrain
        terrain = self.classify_terrain(sensor_data)
        
        # Safety checks
        if sensor_data.ultrasonic_distance < self.min_distance:
            # Emergency stop
            action = ActionSpace.STOP.value
            await self.execute_action(action)
            
            # Then back up
            await self.move_robot("backward", duration=1.0)
        else:
            # Select and execute action
            action = self.select_action(state_tensor)
            await self.execute_action(action)
        
        # Calculate reward if we have previous state
        reward = 0.0
        if self.previous_state is not None:
            reward = self.calculate_reward(
                sensor_data, action, self.previous_state
            )
            
            # Train the model
            self.train_step(
                self.state_to_tensor(self.previous_state, self.control.body_height),
                self.previous_action,
                reward,
                state_tensor,
                False
            )
        
        # Update state
        self.previous_state = sensor_data
        self.previous_action = action
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "action": ActionSpace(action).name,
            "terrain": terrain.value,
            "distance": sensor_data.ultrasonic_distance,
            "reward": reward,
            "epsilon": self.epsilon,
            "battery": sensor_data.battery_voltage
        }
    
    async def run_autonomous_exploration(self, duration_minutes: float = 5.0):
        """Run autonomous exploration for specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = []
        
        while time.time() < end_time:
            try:
                step_result = await self.autonomous_step()
                results.append(step_result)
                
                # Log progress
                if self.step_count % 10 == 0:
                    print(f"Step {self.step_count}: {step_result['action']} "
                          f"(distance: {step_result['distance']:.1f}cm, "
                          f"reward: {step_result['reward']:.2f})")
                
                await asyncio.sleep(0.1)  # Small delay between steps
                
            except Exception as e:
                print(f"Error in autonomous step: {e}")
                await self.execute_action(ActionSpace.STOP.value)
                await asyncio.sleep(1.0)
        
        # Stop robot
        await self.execute_action(ActionSpace.STOP.value)
        
        # Save experience
        self.save_models()
        
        return {
            "duration": time.time() - start_time,
            "steps": self.step_count,
            "average_reward": np.mean([r["reward"] for r in results[-100:]]),
            "final_epsilon": self.epsilon,
            "terrains_encountered": list(set(r["terrain"] for r in results))
        }
    
    def save_models(self):
        """Save trained models"""
        torch.save(self.brain.state_dict(), "hexapod_brain.pth")
        torch.save(self.terrain_classifier.state_dict(), "terrain_classifier.pth")
        
        # Save experience buffer
        with open("experience_buffer.pkl", "wb") as f:
            pickle.dump(self.experience_buffer, f)
    
    def load_models(self):
        """Load pre-trained models if available"""
        try:
            self.brain.load_state_dict(torch.load("hexapod_brain.pth"))
            self.terrain_classifier.load_state_dict(torch.load("terrain_classifier.pth"))
            print("Loaded pre-trained models")
        except:
            print("No pre-trained models found, starting fresh")

# ========== API ENDPOINTS ==========

@app.get("/")
def root():
    """API root endpoint with basic info."""
    return {
        "title": "Enhanced Hexapod Control API with ML",
        "version": "2.0",
        "features": [
            "Smooth servo movements",
            "Natural gait patterns",
            "Body pose control",
            "Movement sequences",
            "Real-time sensor data",
            "ML autonomous navigation"
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

# ========== ML ENDPOINTS ==========

@app.post("/ml/initialize")
async def initialize_ml_controller():
    """Initialize the ML autonomous controller"""
    global ml_controller
    
    sensors = {
        'ultrasonic': ultrasonic,
        'adc': adc,
        'imu': control.imu if hasattr(control, 'imu') else None
    }
    
    ml_controller = AutonomousController(control, servo_controller, sensors)
    
    return {"status": "ML controller initialized", "models_loaded": True}

@app.post("/ml/explore")
async def start_autonomous_exploration(
    background_tasks: BackgroundTasks,
    duration_minutes: float = 5.0
):
    """Start autonomous exploration using ML"""
    if ml_controller is None:
        raise HTTPException(status_code=400, detail="ML controller not initialized")
    
    # Run exploration in background
    background_tasks.add_task(
        ml_controller.run_autonomous_exploration,
        duration_minutes
    )
    
    return {
        "status": "Autonomous exploration started",
        "duration_minutes": duration_minutes
    }

@app.get("/ml/status")
async def get_ml_status():
    """Get current ML controller status"""
    if ml_controller is None:
        return {"status": "Not initialized"}
    
    sensor_data = await ml_controller.get_sensor_state()
    terrain = ml_controller.classify_terrain(sensor_data)
    
    return {
        "status": "Active",
        "steps": ml_controller.step_count,
        "epsilon": ml_controller.epsilon,
        "current_terrain": terrain.value,
        "sensor_data": {
            "distance": sensor_data.ultrasonic_distance,
            "battery": sensor_data.battery_voltage,
            "orientation": {
                "pitch": sensor_data.pitch,
                "roll": sensor_data.roll,
                "yaw": sensor_data.yaw
            }
        }
    }

@app.post("/ml/train")
async def train_ml_model(episodes: int = 10):
    """Train the ML model through simulated episodes"""
    if ml_controller is None:
        raise HTTPException(status_code=400, detail="ML controller not initialized")
    
    results = []
    for episode in range(episodes):
        episode_result = await ml_controller.run_autonomous_exploration(
            duration_minutes=1.0
        )
        results.append(episode_result)
    
    return {
        "episodes_completed": episodes,
        "results": results
    }

@app.post("/ml/step")
async def execute_ml_step():
    """Execute a single ML-controlled step"""
    if ml_controller is None:
        raise HTTPException(status_code=400, detail="ML controller not initialized")
    
    result = await ml_controller.autonomous_step()
    return result

# ========== MAIN ==========

if __name__ == "__main__":
    import uvicorn
    
    # Start the control thread
    control.condition_thread.start()
    
    # Run the FastAPI server
    print("Starting Enhanced Hexapod API Server with ML...")
    print("Access the API at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)