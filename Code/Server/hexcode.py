"""Advanced Hexapod Movement System with IK, Gaits, and Pose Control"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Constants for the hexapod dimensions (adjust these to match your robot)
COXA_LENGTH = 30  # mm - horizontal segment length
FEMUR_LENGTH = 100  # mm - upper leg segment length  
TIBIA_LENGTH = 125  # mm - lower leg segment length

# Servo angle limits
COXA_MIN, COXA_MAX = 0, 170
FEMUR_MIN, FEMUR_MAX = 0, 180
TIBIA_MIN, TIBIA_MAX = 0, 180

# Default positions
DEFAULT_HEIGHT = 100  # mm
DEFAULT_STANCE_RADIUS = 150  # mm

class GaitType(Enum):
    TRIPOD = "tripod"  # Fastest - alternating tripods
    WAVE = "wave"      # Smooth - one leg at a time
    RIPPLE = "ripple"  # Medium - two legs at a time
    

@dataclass
class LegPosition:
    """3D position of leg tip relative to coxa joint"""
    x: float
    y: float
    z: float
    

@dataclass
class LegAngles:
    """Servo angles for a leg"""
    coxa: float
    femur: float
    tibia: float
    

@dataclass
class BodyPose:
    """Robot body pose parameters"""
    x: float = 0        # Forward/backward translation
    y: float = 0        # Left/right translation
    z: float = DEFAULT_HEIGHT  # Height
    roll: float = 0     # Roll angle (degrees)
    pitch: float = 0    # Pitch angle (degrees)
    yaw: float = 0      # Yaw angle (degrees)


class HexapodKinematics:
    """Inverse kinematics solver for hexapod legs"""
    
    def __init__(self):
        # Leg mounting angles (degrees) - where each coxa is mounted
        self.leg_mount_angles = {
            1: -30,   # Front right
            2: -90,   # Middle right
            3: -150,  # Back right
            4: 150,   # Back left
            5: 90,    # Middle left
            6: 30     # Front left
        }
        
        # Initialize leg positions to default stance
        self.leg_positions = {}
        for leg_id in range(1, 7):
            angle_rad = np.radians(self.leg_mount_angles[leg_id])
            x = DEFAULT_STANCE_RADIUS * np.cos(angle_rad)
            y = DEFAULT_STANCE_RADIUS * np.sin(angle_rad)
            self.leg_positions[leg_id] = LegPosition(x, y, -DEFAULT_HEIGHT)
    
    def inverse_kinematics(self, leg_id: int, target: LegPosition) -> Optional[LegAngles]:
        """Calculate servo angles for target position"""
        
        # Get mounting angle for this leg
        mount_angle = np.radians(self.leg_mount_angles[leg_id])
        
        # Transform target to leg coordinate system
        x_leg = target.x * np.cos(-mount_angle) - target.y * np.sin(-mount_angle)
        y_leg = target.x * np.sin(-mount_angle) + target.y * np.cos(-mount_angle)
        z_leg = target.z
        
        # Calculate coxa angle
        coxa_angle = np.degrees(np.arctan2(y_leg, x_leg))
        
        # Calculate distance from coxa to target in horizontal plane
        horizontal_dist = np.sqrt(x_leg**2 + y_leg**2) - COXA_LENGTH
        
        # Check if target is reachable
        total_dist = np.sqrt(horizontal_dist**2 + z_leg**2)
        if total_dist > (FEMUR_LENGTH + TIBIA_LENGTH):
            return None  # Target unreachable
        
        # Calculate femur and tibia angles using law of cosines
        try:
            # Angle between femur and horizontal
            femur_ground_angle = np.arctan2(-z_leg, horizontal_dist)
            
            # Internal angle of triangle formed by femur, tibia, and line to target
            cos_internal = (FEMUR_LENGTH**2 + total_dist**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * total_dist)
            cos_internal = np.clip(cos_internal, -1, 1)
            internal_angle = np.arccos(cos_internal)
            
            femur_angle = np.degrees(femur_ground_angle + internal_angle)
            
            # Knee angle
            cos_knee = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - total_dist**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
            cos_knee = np.clip(cos_knee, -1, 1)
            knee_angle = np.arccos(cos_knee)
            
            tibia_angle = np.degrees(np.pi - knee_angle)
            
        except ValueError:
            return None  # Math error, target unreachable
        
        # Adjust angles to servo orientation and limits
        coxa_servo = 90 + coxa_angle
        femur_servo = 90 - femur_angle
        tibia_servo = 180 - tibia_angle  # Inverse for tibia
        
        # Clamp to servo limits
        coxa_servo = np.clip(coxa_servo, COXA_MIN, COXA_MAX)
        femur_servo = np.clip(femur_servo, FEMUR_MIN, FEMUR_MAX)
        tibia_servo = np.clip(tibia_servo, TIBIA_MIN, TIBIA_MAX)
        
        return LegAngles(coxa_servo, femur_servo, tibia_servo)
    
    def apply_body_transform(self, pose: BodyPose) -> Dict[int, LegPosition]:
        """Apply body pose transformation to all legs"""
        transformed_positions = {}
        
        # Convert angles to radians
        roll_rad = np.radians(pose.roll)
        pitch_rad = np.radians(pose.pitch)
        yaw_rad = np.radians(pose.yaw)
        
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        for leg_id, leg_pos in self.leg_positions.items():
            # Get leg attachment point
            mount_angle = np.radians(self.leg_mount_angles[leg_id])
            attach_x = COXA_LENGTH * np.cos(mount_angle)
            attach_y = COXA_LENGTH * np.sin(mount_angle)
            attach_z = 0
            
            # Apply rotation to attachment point
            attach_point = np.array([attach_x, attach_y, attach_z])
            rotated_attach = R @ attach_point
            
            # Calculate new leg position
            new_x = leg_pos.x + pose.x - (rotated_attach[0] - attach_x)
            new_y = leg_pos.y + pose.y - (rotated_attach[1] - attach_y)
            new_z = leg_pos.z - pose.z - (rotated_attach[2] - attach_z)
            
            transformed_positions[leg_id] = LegPosition(new_x, new_y, new_z)
        
        return transformed_positions


class GaitGenerator:
    """Generate smooth gait patterns for hexapod walking"""
    
    def __init__(self, kinematics: HexapodKinematics):
        self.kinematics = kinematics
        self.gait_sequences = {
            GaitType.TRIPOD: [
                [1, 3, 5],  # Group 1: Front-right, back-right, middle-left
                [2, 4, 6]   # Group 2: Middle-right, back-left, front-left
            ],
            GaitType.WAVE: [
                [1], [6], [2], [5], [3], [4]  # One leg at a time
            ],
            GaitType.RIPPLE: [
                [1, 4], [2, 5], [3, 6]  # Opposite pairs
            ]
        }
        
    def generate_step_positions(self, direction: Tuple[float, float], 
                               step_length: float, step_height: float,
                               gait_type: GaitType, phase: float) -> Dict[int, LegPosition]:
        """Generate leg positions for a single step phase"""
        
        positions = {}
        sequence = self.gait_sequences[gait_type]
        
        # Normalize direction
        dir_mag = np.sqrt(direction[0]**2 + direction[1]**2)
        if dir_mag > 0:
            dir_norm = (direction[0] / dir_mag, direction[1] / dir_mag)
        else:
            dir_norm = (0, 0)
        
        # Calculate which legs are in swing vs stance phase
        group_count = len(sequence)
        group_phase = phase * group_count
        current_group = int(group_phase) % group_count
        group_progress = group_phase - int(group_phase)
        
        for leg_id in range(1, 7):
            base_pos = self.kinematics.leg_positions[leg_id]
            
            # Check which group this leg belongs to
            leg_group = None
            for i, group in enumerate(sequence):
                if leg_id in group:
                    leg_group = i
                    break
            
            if leg_group == current_group:
                # Swing phase - leg in air moving forward
                swing_progress = group_progress
                
                # Parabolic trajectory for smooth motion
                x_offset = step_length * dir_norm[0] * (2 * swing_progress - 1)
                y_offset = step_length * dir_norm[1] * (2 * swing_progress - 1)
                z_offset = step_height * 4 * swing_progress * (1 - swing_progress)
                
                positions[leg_id] = LegPosition(
                    base_pos.x + x_offset,
                    base_pos.y + y_offset,
                    base_pos.z + z_offset
                )
            else:
                # Stance phase - leg on ground moving backward
                stance_progress = (group_phase - leg_group) / (group_count - 1)
                stance_progress = stance_progress % 1.0
                
                x_offset = -step_length * dir_norm[0] * stance_progress
                y_offset = -step_length * dir_norm[1] * stance_progress
                
                positions[leg_id] = LegPosition(
                    base_pos.x + x_offset,
                    base_pos.y + y_offset,
                    base_pos.z
                )
        
        return positions


class SmoothMotionController:
    """Handles smooth interpolation between positions"""
    
    def __init__(self, servo_controller):
        self.servo = servo_controller
        self.current_angles = {i: 90 for i in range(32)}  # Track current positions
        self.leg_servo_map = {
            1: [15, 14, 13],
            2: [12, 11, 10],
            3: [9, 8, 31],
            4: [22, 23, 27],
            5: [19, 20, 21],
            6: [16, 17, 18]
        }
        
    def interpolate_move(self, target_angles: Dict[int, float], duration: float, steps: int = 50):
        """Smoothly interpolate to target angles"""
        
        start_angles = self.current_angles.copy()
        
        for step in range(steps + 1):
            t = step / steps  # Progress from 0 to 1
            
            # Use sinusoidal interpolation for smoother acceleration/deceleration
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            
            for channel, target in target_angles.items():
                if channel in start_angles:
                    current = start_angles[channel]
                    interpolated = current + (target - current) * t_smooth
                    self.servo.set_servo_angle(channel, int(interpolated))
                    self.current_angles[channel] = interpolated
            
            time.sleep(duration / steps)
    
    def set_leg_angles(self, leg_id: int, angles: LegAngles, duration: float = 0.5):
        """Set all angles for a specific leg with smooth interpolation"""
        
        servos = self.leg_servo_map[leg_id]
        target_angles = {
            servos[0]: angles.coxa,
            servos[1]: angles.femur,
            servos[2]: angles.tibia
        }
        
        self.interpolate_move(target_angles, duration)
    
    def set_all_legs(self, leg_angles: Dict[int, LegAngles], duration: float = 0.5):
        """Set angles for all legs simultaneously"""
        
        target_angles = {}
        for leg_id, angles in leg_angles.items():
            servos = self.leg_servo_map[leg_id]
            target_angles[servos[0]] = angles.coxa
            target_angles[servos[1]] = angles.femur
            target_angles[servos[2]] = angles.tibia
        
        self.interpolate_move(target_angles, duration)


class HexapodController:
    """Main controller combining all systems"""
    
    def __init__(self, servo_controller):
        self.servo = servo_controller
        self.kinematics = HexapodKinematics()
        self.gait_gen = GaitGenerator(self.kinematics)
        self.motion = SmoothMotionController(servo_controller)
        self.current_pose = BodyPose()
        self.is_walking = False
        
    def set_body_pose(self, pose: BodyPose, duration: float = 1.0):
        """Set robot body pose (height, tilt, etc.)"""
        
        self.current_pose = pose
        
        # Apply body transformation
        transformed_positions = self.kinematics.apply_body_transform(pose)
        
        # Calculate IK for all legs
        leg_angles = {}
        for leg_id, position in transformed_positions.items():
            angles = self.kinematics.inverse_kinematics(leg_id, position)
            if angles:
                leg_angles[leg_id] = angles
        
        # Apply smooth motion
        self.motion.set_all_legs(leg_angles, duration)
    
    def stand_up(self, height: float = DEFAULT_HEIGHT, duration: float = 2.0):
        """Stand up to specified height"""
        
        pose = BodyPose(z=height)
        self.set_body_pose(pose, duration)
    
    def walk(self, direction: Tuple[float, float], speed: float = 1.0, 
             gait: GaitType = GaitType.TRIPOD):
        """Walk in specified direction with given gait"""
        
        self.is_walking = True
        step_length = 40 * speed  # mm
        step_height = 30  # mm
        cycle_time = 2.0 / speed  # seconds
        
        phase = 0.0
        while self.is_walking:
            # Generate step positions
            step_positions = self.gait_gen.generate_step_positions(
                direction, step_length, step_height, gait, phase
            )
            
            # Apply body pose transformation
            transformed_positions = {}
            for leg_id, leg_pos in step_positions.items():
                # Apply current body pose
                mount_angle = np.radians(self.kinematics.leg_mount_angles[leg_id])
                attach_x = COXA_LENGTH * np.cos(mount_angle)
                attach_y = COXA_LENGTH * np.sin(mount_angle)
                
                transformed_positions[leg_id] = LegPosition(
                    leg_pos.x + self.current_pose.x,
                    leg_pos.y + self.current_pose.y,
                    leg_pos.z - self.current_pose.z
                )
            
            # Calculate IK
            leg_angles = {}
            for leg_id, position in transformed_positions.items():
                angles = self.kinematics.inverse_kinematics(leg_id, position)
                if angles:
                    leg_angles[leg_id] = angles
            
            # Apply motion
            self.motion.set_all_legs(leg_angles, cycle_time / 20)
            
            # Update phase
            phase = (phase + 0.05) % 1.0
            time.sleep(cycle_time / 20)
    
    def stop_walking(self):
        """Stop walking and return to neutral stance"""
        self.is_walking = False
        time.sleep(0.1)
        self.stand_up()
    
    def tilt_demo(self):
        """Demonstrate tilting capabilities"""
        
        # Tilt forward
        self.set_body_pose(BodyPose(pitch=15), 1.0)
        time.sleep(1)
        
        # Tilt back
        self.set_body_pose(BodyPose(pitch=-15), 1.0)
        time.sleep(1)
        
        # Roll left
        self.set_body_pose(BodyPose(roll=15), 1.0)
        time.sleep(1)
        
        # Roll right
        self.set_body_pose(BodyPose(roll=-15), 1.0)
        time.sleep(1)
        
        # Return to neutral
        self.set_body_pose(BodyPose(), 1.0)
    
    def dance_demo(self):
        """Fun dance demonstration"""
        
        # Wave motion
        for i in range(3):
            self.set_body_pose(BodyPose(z=60, roll=10), 0.5)
            self.set_body_pose(BodyPose(z=100, roll=-10), 0.5)
        
        # Circular motion
        for angle in range(0, 360, 30):
            x = 20 * np.cos(np.radians(angle))
            y = 20 * np.sin(np.radians(angle))
            self.set_body_pose(BodyPose(x=x, y=y, z=80), 0.3)
        
        # Return to neutral
        self.set_body_pose(BodyPose(), 1.0)


# FastAPI integration models
class PoseCommand(BaseModel):
    x: float = 0
    y: float = 0
    z: float = DEFAULT_HEIGHT
    roll: float = 0
    pitch: float = 0
    yaw: float = 0
    duration: float = 1.0


class WalkCommand(BaseModel):
    direction_x: float
    direction_y: float
    speed: float = 1.0
    gait: str = "tripod"


class StandCommand(BaseModel):
    height: float = DEFAULT_HEIGHT
    duration: float = 2.0


# Add these endpoints to your existing FastAPI app
def add_advanced_endpoints(app: FastAPI, hexapod: HexapodController):
    
    @app.post("/advanced/pose")
    async def set_pose(cmd: PoseCommand):
        """Set robot body pose"""
        pose = BodyPose(
            x=cmd.x, y=cmd.y, z=cmd.z,
            roll=cmd.roll, pitch=cmd.pitch, yaw=cmd.yaw
        )
        hexapod.set_body_pose(pose, cmd.duration)
        return {"status": "pose set"}
    
    @app.post("/advanced/stand")
    async def stand(cmd: StandCommand):
        """Stand up to specified height"""
        hexapod.stand_up(cmd.height, cmd.duration)
        return {"status": "standing"}
    
    @app.post("/advanced/walk/start")
    async def start_walking(cmd: WalkCommand):
        """Start walking in specified direction"""
        gait_map = {
            "tripod": GaitType.TRIPOD,
            "wave": GaitType.WAVE,
            "ripple": GaitType.RIPPLE
        }
        gait = gait_map.get(cmd.gait, GaitType.TRIPOD)
        
        # Run walking in background
        asyncio.create_task(asyncio.to_thread(
            hexapod.walk, 
            (cmd.direction_x, cmd.direction_y), 
            cmd.speed, 
            gait
        ))
        return {"status": "walking started"}
    
    @app.post("/advanced/walk/stop")
    async def stop_walking():
        """Stop walking"""
        hexapod.stop_walking()
        return {"status": "walking stopped"}
    
    @app.post("/advanced/demo/tilt")
    async def tilt_demo():
        """Run tilt demonstration"""
        asyncio.create_task(asyncio.to_thread(hexapod.tilt_demo))
        return {"status": "tilt demo started"}
    
    @app.post("/advanced/demo/dance")
    async def dance_demo():
        """Run dance demonstration"""
        asyncio.create_task(asyncio.to_thread(hexapod.dance_demo))
        return {"status": "dance demo started"}


# Example usage
if __name__ == "__main__":
    from servo import Servo
    
    # Initialize controllers
    servo = Servo()
    hexapod = HexapodController(servo)
    
    # Stand up
    print("Standing up...")
    hexapod.stand_up()
    time.sleep(2)
    
    # Walk forward
    print("Walking forward...")
    hexapod.walk((1, 0), speed=1.0, gait=GaitType.TRIPOD)