"""
Object detection integration with action prediction for drone VLA system
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: float


@dataclass
class ObjectAction:
    """Action recommendation based on detected object"""
    object_id: str
    action_type: str  # 'approach', 'avoid', 'follow', 'land_near', 'inspect'
    target_position: Tuple[float, float, float]  # relative position
    urgency: float  # 0-1 scale
    description: str


class ObjectActionMapper:
    """Maps detected objects to appropriate drone actions"""
    
    def __init__(self):
        # Define object-action mappings
        self.action_rules = {
            # Objects to approach
            'person': {'action': 'follow', 'distance': 3.0, 'urgency': 0.7},
            'car': {'action': 'inspect', 'distance': 5.0, 'urgency': 0.5},
            'bicycle': {'action': 'follow', 'distance': 2.0, 'urgency': 0.6},
            
            # Landing targets
            'landing_pad': {'action': 'land_near', 'distance': 0.5, 'urgency': 0.9},
            'helipad': {'action': 'land_near', 'distance': 1.0, 'urgency': 0.8},
            
            # Objects to avoid
            'bird': {'action': 'avoid', 'distance': 10.0, 'urgency': 0.9},
            'airplane': {'action': 'avoid', 'distance': 50.0, 'urgency': 1.0},
            'helicopter': {'action': 'avoid', 'distance': 30.0, 'urgency': 0.95},
            'tree': {'action': 'avoid', 'distance': 5.0, 'urgency': 0.8},
            'building': {'action': 'avoid', 'distance': 10.0, 'urgency': 0.7},
            
            # Objects to inspect
            'fire': {'action': 'inspect', 'distance': 15.0, 'urgency': 0.8},
            'smoke': {'action': 'inspect', 'distance': 10.0, 'urgency': 0.7},
            'traffic_light': {'action': 'inspect', 'distance': 5.0, 'urgency': 0.4},
        }
        
        # Default action for unknown objects
        self.default_action = {'action': 'inspect', 'distance': 5.0, 'urgency': 0.3}
        
    def map_objects_to_actions(self, 
                              objects: List[DetectedObject],
                              frame_shape: Tuple[int, int]) -> List[ObjectAction]:
        """
        Map detected objects to recommended actions
        
        Args:
            objects: List of detected objects
            frame_shape: (height, width) of the frame
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        for i, obj in enumerate(objects):
            # Get action rule for this object
            rule = self.action_rules.get(obj.class_name, self.default_action)
            
            # Calculate relative position
            relative_pos = self._calculate_relative_position(obj, frame_shape, rule['distance'])
            
            # Create action recommendation
            action = ObjectAction(
                object_id=f"{obj.class_name}_{i}",
                action_type=rule['action'],
                target_position=relative_pos,
                urgency=rule['urgency'] * obj.confidence,
                description=f"{rule['action'].replace('_', ' ').title()} {obj.class_name}"
            )
            
            actions.append(action)
            
        # Sort by urgency (highest first)
        actions.sort(key=lambda x: x.urgency, reverse=True)
        
        return actions
        
    def _calculate_relative_position(self, 
                                   obj: DetectedObject, 
                                   frame_shape: Tuple[int, int],
                                   target_distance: float) -> Tuple[float, float, float]:
        """Calculate relative position for action target"""
        height, width = frame_shape
        
        # Convert pixel coordinates to normalized coordinates
        center_x, center_y = obj.center
        norm_x = (center_x - width / 2) / (width / 2)   # -1 to 1
        norm_y = (center_y - height / 2) / (height / 2) # -1 to 1
        
        # Estimate 3D position (simplified model)
        # Larger objects are assumed to be closer
        size_factor = min(obj.area / (width * height), 0.5)
        estimated_distance = target_distance / (size_factor + 0.1)
        
        # Convert to relative drone coordinates
        # Forward: positive Y, Right: positive X, Down: positive Z
        relative_x = norm_x * estimated_distance * 0.5  # Side movement
        relative_y = estimated_distance                  # Forward distance
        relative_z = -norm_y * estimated_distance * 0.3 # Altitude adjustment
        
        return (relative_x, relative_y, relative_z)


class VisionActionIntegrator:
    """Integrates object detection with VLA model for action generation"""
    
    def __init__(self, vla_model, object_detector=None):
        self.vla_model = vla_model
        self.object_detector = object_detector
        self.object_mapper = ObjectActionMapper()
        
        # If no detector provided, use built-in YOLOv8
        if object_detector is None:
            from ..core.camera import ObjectDetector
            self.object_detector = ObjectDetector()
        
    def process_frame_with_objects(self, 
                                  frames: List[np.ndarray],
                                  text_command: str,
                                  detection_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process frames with object detection and generate actions
        
        Args:
            frames: List of video frames
            text_command: Text command from user
            detection_threshold: Confidence threshold for object detection
            
        Returns:
            Dictionary with actions, objects, and metadata
        """
        result = {
            'vla_action': None,
            'object_actions': [],
            'detected_objects': [],
            'final_action': None,
            'reasoning': []
        }
        
        # Get VLA model prediction
        vla_actions, vla_confidence = self.vla_model.predict_action(frames, text_command)
        result['vla_action'] = vla_actions[0] if len(vla_actions) > 0 else np.zeros(6)
        
        # Detect objects in the latest frame
        latest_frame = frames[-1]
        detections = self.object_detector.detect_objects(latest_frame, detection_threshold)
        
        # Convert detections to DetectedObject format
        detected_objects = []
        for det in detections:
            bbox = det['bbox']
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            obj = DetectedObject(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=tuple(map(int, bbox)),
                center=center,
                area=area
            )
            detected_objects.append(obj)
            
        result['detected_objects'] = detected_objects
        
        # Map objects to actions
        if detected_objects:
            object_actions = self.object_mapper.map_objects_to_actions(
                detected_objects, latest_frame.shape[:2]
            )
            result['object_actions'] = object_actions
            
            # Integrate object-based actions with VLA actions
            final_action = self._integrate_actions(
                result['vla_action'], object_actions, text_command
            )
            result['final_action'] = final_action
            result['reasoning'] = self._generate_reasoning(object_actions, text_command)
        else:
            result['final_action'] = result['vla_action']
            result['reasoning'] = ["No objects detected, using VLA model prediction"]
            
        return result
        
    def _integrate_actions(self, 
                          vla_action: np.ndarray,
                          object_actions: List[ObjectAction],
                          text_command: str) -> np.ndarray:
        """Integrate VLA action with object-based actions"""
        if not object_actions:
            return vla_action
            
        # Start with VLA action as base
        integrated_action = vla_action.copy()
        
        # Apply highest priority object action
        primary_action = object_actions[0]
        
        # Modify action based on object type
        if primary_action.action_type == 'avoid':
            # Move away from object
            avoid_factor = primary_action.urgency
            integrated_action[0] -= primary_action.target_position[0] * avoid_factor
            integrated_action[1] -= primary_action.target_position[1] * avoid_factor * 0.5
            integrated_action[2] += abs(primary_action.target_position[2]) * avoid_factor
            
        elif primary_action.action_type in ['approach', 'follow']:
            # Move towards object
            approach_factor = primary_action.urgency * 0.5
            integrated_action[0] += primary_action.target_position[0] * approach_factor
            integrated_action[1] += primary_action.target_position[1] * approach_factor
            
        elif primary_action.action_type == 'land_near':
            # Override with landing action
            integrated_action[0] = primary_action.target_position[0] * 0.3
            integrated_action[1] = primary_action.target_position[1] * 0.3
            integrated_action[2] = -0.5  # Descend
            
        elif primary_action.action_type == 'inspect':
            # Careful approach
            inspect_factor = primary_action.urgency * 0.3
            integrated_action[0] += primary_action.target_position[0] * inspect_factor
            integrated_action[1] += primary_action.target_position[1] * inspect_factor
            integrated_action[5] = 0.2  # Slow rotation for inspection
            
        # Ensure action bounds
        integrated_action = np.clip(integrated_action, -1.0, 1.0)
        
        return integrated_action
        
    def _generate_reasoning(self, 
                          object_actions: List[ObjectAction],
                          text_command: str) -> List[str]:
        """Generate human-readable reasoning for actions"""
        reasoning = []
        
        if object_actions:
            primary = object_actions[0]
            reasoning.append(f"Primary object: {primary.object_id}")
            reasoning.append(f"Recommended action: {primary.description}")
            reasoning.append(f"Urgency level: {primary.urgency:.2f}")
            
            if len(object_actions) > 1:
                reasoning.append(f"Additional objects detected: {len(object_actions) - 1}")
                
        reasoning.append(f"User command: '{text_command}'")
        
        return reasoning


def create_vision_action_integrator(vla_model, object_detector=None):
    """Factory function to create vision-action integrator"""
    return VisionActionIntegrator(vla_model, object_detector)