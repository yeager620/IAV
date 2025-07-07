"""
Command validation and safety checks for drone VLA system
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for different operational modes"""
    STRICT = "strict"      # Maximum safety, minimal risk
    NORMAL = "normal"      # Balanced safety and performance
    RELAXED = "relaxed"    # Reduced safety for advanced users


@dataclass
class SafetyLimits:
    """Safety limits for drone operations"""
    max_velocity: float = 2.0          # m/s
    max_acceleration: float = 1.0      # m/s²
    max_altitude: float = 30.0         # meters
    min_altitude: float = 0.5          # meters
    max_angular_velocity: float = 1.0  # rad/s
    min_battery: float = 20.0          # percent
    max_tilt_angle: float = 30.0       # degrees
    geofence_radius: float = 100.0     # meters from origin


class CommandValidator:
    """Validates text commands for safety and feasibility"""
    
    def __init__(self):
        self.dangerous_keywords = {
            'crash', 'destroy', 'attack', 'collide', 'ram', 'damage',
            'break', 'smash', 'hit', 'strike', 'bomb', 'weapon'
        }
        
        self.restricted_areas = {
            'airport', 'airfield', 'runway', 'helipad', 'military',
            'prison', 'nuclear', 'power plant', 'hospital', 'school'
        }
        
    def validate_command(self, command: str) -> Tuple[bool, str]:
        """
        Validate text command for safety
        
        Args:
            command: Text command to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not command or not isinstance(command, str):
            return False, "Invalid command format"
            
        command_lower = command.lower()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in command_lower:
                return False, f"Dangerous keyword detected: {keyword}"
                
        # Check for restricted areas
        for area in self.restricted_areas:
            if area in command_lower:
                return False, f"Restricted area mentioned: {area}"
                
        # Check command length
        if len(command) > 200:
            return False, "Command too long"
            
        return True, "Command validated"


class ActionValidator:
    """Validates action vectors for safety constraints"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.NORMAL):
        self.safety_level = safety_level
        self.limits = SafetyLimits()
        
        # Adjust limits based on safety level
        if safety_level == SafetyLevel.STRICT:
            self.limits.max_velocity *= 0.5
            self.limits.max_acceleration *= 0.5
            self.limits.max_angular_velocity *= 0.5
        elif safety_level == SafetyLevel.RELAXED:
            self.limits.max_velocity *= 1.5
            self.limits.max_acceleration *= 1.5
            
    def validate_action(self, 
                       action: np.ndarray,
                       current_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Validate and constrain action vector
        
        Args:
            action: Action vector [vx, vy, vz, roll_rate, pitch_rate, yaw_rate]
            current_state: Current drone state for context-aware validation
            
        Returns:
            Tuple of (constrained_action, warnings)
        """
        if len(action) != 6:
            raise ValueError("Action must be 6-dimensional vector")
            
        constrained_action = action.copy()
        warnings = []
        
        # Velocity constraints
        velocity = constrained_action[:3]
        velocity_magnitude = np.linalg.norm(velocity)
        
        if velocity_magnitude > self.limits.max_velocity:
            scale_factor = self.limits.max_velocity / velocity_magnitude
            constrained_action[:3] *= scale_factor
            warnings.append(f"Velocity scaled down by {scale_factor:.2f}")
            
        # Angular velocity constraints
        angular_velocity = constrained_action[3:]
        angular_velocity_magnitude = np.linalg.norm(angular_velocity)
        
        if angular_velocity_magnitude > self.limits.max_angular_velocity:
            scale_factor = self.limits.max_angular_velocity / angular_velocity_magnitude
            constrained_action[3:] *= scale_factor
            warnings.append(f"Angular velocity scaled down by {scale_factor:.2f}")
            
        # Context-aware validation
        if current_state:
            constrained_action, state_warnings = self._validate_with_state(
                constrained_action, current_state
            )
            warnings.extend(state_warnings)
            
        return constrained_action, warnings
        
    def _validate_with_state(self, 
                           action: np.ndarray, 
                           state: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Validate action with current state context"""
        constrained_action = action.copy()
        warnings = []
        
        # Altitude constraints
        if 'altitude' in state:
            altitude = state['altitude']
            vertical_velocity = constrained_action[2]
            
            # Prevent flying too high
            if altitude > self.limits.max_altitude and vertical_velocity > 0:
                constrained_action[2] = 0
                warnings.append("Blocked upward motion: altitude limit reached")
                
            # Prevent flying too low
            if altitude < self.limits.min_altitude and vertical_velocity < 0:
                constrained_action[2] = 0
                warnings.append("Blocked downward motion: minimum altitude reached")
                
        # Battery constraints
        if 'battery' in state:
            battery = state['battery']
            if battery < self.limits.min_battery:
                # Force landing if battery is critically low
                constrained_action = np.array([0, 0, -0.5, 0, 0, 0])
                warnings.append("Emergency landing: low battery")
                
        # Geofence constraints
        if 'position' in state:
            position = state['position']
            distance_from_origin = np.linalg.norm(position[:2])
            
            if distance_from_origin > self.limits.geofence_radius:
                # Redirect towards origin
                direction_to_origin = -position[:2] / np.linalg.norm(position[:2])
                constrained_action[:2] = direction_to_origin * 0.5
                warnings.append("Redirected: outside geofence")
                
        return constrained_action, warnings


class ConfidenceFilter:
    """Filters actions based on model confidence"""
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        
    def filter_action(self, 
                     action: np.ndarray, 
                     confidence: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filter action based on confidence scores
        
        Args:
            action: Action vector
            confidence: Confidence scores for each action dimension
            
        Returns:
            Tuple of (filtered_action, is_reliable)
        """
        # Check overall confidence
        avg_confidence = np.mean(confidence)
        
        if avg_confidence < self.min_confidence:
            # Return conservative action (hover)
            conservative_action = np.zeros_like(action)
            return conservative_action, False
            
        # Scale action by confidence
        filtered_action = action * confidence
        
        # Check individual dimension confidence
        low_confidence_dims = confidence < self.min_confidence
        if np.any(low_confidence_dims):
            filtered_action[low_confidence_dims] = 0
            
        return filtered_action, True


class SafetyMonitor:
    """Comprehensive safety monitoring system"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.NORMAL):
        self.command_validator = CommandValidator()
        self.action_validator = ActionValidator(safety_level)
        self.confidence_filter = ConfidenceFilter()
        self.emergency_stop = False
        
    def validate_and_filter(self, 
                          command: str,
                          action: np.ndarray,
                          confidence: np.ndarray,
                          current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete validation and filtering pipeline
        
        Args:
            command: Text command
            action: Predicted action vector
            confidence: Action confidence scores
            current_state: Current drone state
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'safe_action': np.zeros(6),
            'is_safe': False,
            'warnings': [],
            'blocked_reason': None
        }
        
        # Emergency stop check
        if self.emergency_stop:
            result['blocked_reason'] = "Emergency stop activated"
            return result
            
        # Validate command
        command_valid, command_reason = self.command_validator.validate_command(command)
        if not command_valid:
            result['blocked_reason'] = f"Command validation failed: {command_reason}"
            return result
            
        # Filter by confidence
        filtered_action, is_reliable = self.confidence_filter.filter_action(action, confidence)
        if not is_reliable:
            result['warnings'].append("Low confidence action filtered")
            
        # Validate action constraints
        safe_action, action_warnings = self.action_validator.validate_action(
            filtered_action, current_state
        )
        result['warnings'].extend(action_warnings)
        
        result['safe_action'] = safe_action
        result['is_safe'] = True
        
        return result
        
    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        logger.warning("Emergency stop activated")
        
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop = False
        logger.info("Emergency stop deactivated")


def create_safety_monitor(safety_level: str = "normal") -> SafetyMonitor:
    """Factory function to create safety monitor"""
    level = SafetyLevel(safety_level.lower())
    return SafetyMonitor(level)