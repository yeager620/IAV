import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Optional, Tuple, List
import threading
import queue
import time
from dataclasses import dataclass


@dataclass
class CameraFrame:
    image: np.ndarray
    timestamp: float
    frame_id: int
    metadata: dict


class CameraInput:
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.frame_id = 0
        self.capture_thread = None
        

        self.camera_matrix = None
        self.dist_coeffs = None
        
    def initialize(self) -> bool:
        """Initialize camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                return False
                

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            

            ret, frame = self.cap.read()
            if not ret:
                return False
                
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
            
    def start_capture(self):
        """Start continuous frame capture in separate thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def stop_capture(self):
        """Stop frame capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    

                camera_frame = CameraFrame(
                    image=frame,
                    timestamp=time.time(),
                    frame_id=self.frame_id,
                    metadata={}
                )
                
                self.frame_id += 1
                

                try:
                    self.frame_queue.put(camera_frame, block=False)
                except queue.Full:

                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(camera_frame, block=False)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                print(f"Error in capture loop: {e}")
                break
                
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_frame(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """Get frame with timeout"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def cleanup(self):
        """Clean up camera resources"""
        self.stop_capture()
        if self.cap:
            self.cap.release()


class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        

        self.viz_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        
    def preprocess_frame(self, frame: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Preprocess frame for neural network input"""
        try:
            # Use Mojo bridge for high-performance preprocessing
            from ..python.mojo_bridge import mojo_camera_process
            
            # Apply Mojo preprocessing if available
            processed_frame = mojo_camera_process(frame)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            if normalize:
                tensor = self.transforms(rgb_frame)
            else:
                tensor = self.viz_transforms(rgb_frame)
                
            return tensor
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            # Fallback to basic processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.transforms(rgb_frame) if normalize else self.viz_transforms(rgb_frame)
            return tensor
        
    def batch_preprocess(self, frames: List[np.ndarray], normalize: bool = True) -> torch.Tensor:
        """Preprocess multiple frames into batch"""
        tensors = []
        for frame in frames:
            tensor = self.preprocess_frame(frame, normalize)
            tensors.append(tensor)
            
        return torch.stack(tensors, dim=0)
        
    def postprocess_output(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to displayable image"""

        denorm = tensor.clone()
        denorm = denorm * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = denorm + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        

        denorm = torch.clamp(denorm, 0, 1)
        np_img = denorm.permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        

        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        return bgr_img


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n"):
        from ultralytics import YOLO
        import os
        
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", f"{model_name}.pt")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[dict]:
        """Detect objects in frame"""
        try:
            results = self.model(frame, conf=confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        
            return detections
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
            
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = map(int, bbox)
            

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                       
        return annotated_frame


class VisionSystem:
    def __init__(self, camera_id: int = 0, target_size: Tuple[int, int] = (224, 224)):
        self.camera = CameraInput(camera_id)
        self.processor = ImageProcessor(target_size)
        self.detector = ObjectDetector()
        
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the vision system"""
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return False
            
        self.camera.start_capture()
        self.is_initialized = True
        return True
        
    def get_processed_frame(self, timeout: float = 1.0) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        """Get processed frame ready for neural network"""
        if not self.is_initialized:
            return None
            
        camera_frame = self.camera.get_frame(timeout)
        if camera_frame is None:
            return None
            

        processed_tensor = self.processor.preprocess_frame(camera_frame.image)
        
        return processed_tensor, camera_frame.image
        
    def detect_and_process(self, timeout: float = 1.0) -> Optional[Tuple[torch.Tensor, List[dict], np.ndarray]]:
        """Get processed frame with object detections"""
        result = self.get_processed_frame(timeout)
        if result is None:
            return None
            
        processed_tensor, raw_frame = result
        

        detections = self.detector.detect_objects(raw_frame)
        
        return processed_tensor, detections, raw_frame
        
    def cleanup(self):
        """Clean up vision system"""
        self.camera.cleanup()