"""
Vision pipeline for processing camera input and preparing frames for VLA model.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Tuple
import threading
import queue
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    frames: np.ndarray
    timestamp: float
    frame_ids: List[int]

class VisionPipeline:
    """Real-time vision processing pipeline for VLA system"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Camera settings
        self.camera_id = config.get('camera_id', 0)
        self.frame_width = config.get('frame_width', 640)
        self.frame_height = config.get('frame_height', 480)
        self.fps = config.get('fps', 30)
        
        # VLA model input settings
        self.target_width = config.get('target_width', 224)
        self.target_height = config.get('target_height', 224)
        self.sequence_length = config.get('sequence_length', 16)
        
        # Camera and processing
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer = queue.Queue(maxsize=self.sequence_length * 2)
        self.processed_frames = queue.Queue(maxsize=5)
        
        # Threading
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
        # Frame tracking
        self.frame_count = 0
        self.last_sequence_time = 0
        
        # Performance tracking
        self.capture_fps = 0
        self.process_fps = 0
        self.last_fps_update = time.time()
        self.fps_frame_count = 0
    
    def initialize(self) -> bool:
        """Initialize camera and start processing threads"""
        try:
            logger.info(f"Initializing camera {self.camera_id}")
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Camera initialized: {test_frame.shape}")
            
            # Start processing threads
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            
            logger.info("Vision pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Vision pipeline initialization failed: {e}")
            return False
    
    async def get_latest_frames(self) -> Optional[np.ndarray]:
        """Get the latest processed frame sequence for VLA model"""
        try:
            frame_data = self.processed_frames.get_nowait()
            return frame_data.frames
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> dict:
        """Get vision pipeline performance statistics"""
        return {
            'capture_fps': self.capture_fps,
            'process_fps': self.process_fps,
            'frame_buffer_size': self.frame_buffer.qsize(),
            'processed_buffer_size': self.processed_frames.qsize(),
            'total_frames': self.frame_count
        }
    
    def _capture_loop(self):
        """Main camera capture loop"""
        logger.info("Starting camera capture loop")
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.01)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                # Add timestamp and frame ID
                timestamp = time.time()
                frame_id = self.frame_count
                self.frame_count += 1
                
                # Add to buffer (drop oldest if full)
                try:
                    self.frame_buffer.put_nowait((frame, timestamp, frame_id))
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((frame, timestamp, frame_id))
                    except queue.Empty:
                        pass
                
                # Update FPS counter
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.capture_fps = self.fps_frame_count / (current_time - self.last_fps_update)
                    self.fps_frame_count = 0
                    self.last_fps_update = current_time
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.01)
    
    def _process_loop(self):
        """Frame processing loop for VLA model preparation"""
        logger.info("Starting frame processing loop")
        frame_sequence = []
        process_frame_count = 0
        last_process_fps_update = time.time()
        
        while self.running:
            try:
                # Get frame from buffer
                try:
                    frame, timestamp, frame_id = self.frame_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Add to sequence
                frame_sequence.append((processed_frame, timestamp, frame_id))
                
                # Maintain sequence length
                if len(frame_sequence) > self.sequence_length:
                    frame_sequence.pop(0)
                
                # Create frame sequence for VLA model when we have enough frames
                if len(frame_sequence) == self.sequence_length:
                    # Stack frames into tensor format
                    frames_array = np.stack([f[0] for f in frame_sequence], axis=0)
                    frame_ids = [f[2] for f in frame_sequence]
                    latest_timestamp = frame_sequence[-1][1]
                    
                    frame_data = FrameData(
                        frames=frames_array,
                        timestamp=latest_timestamp,
                        frame_ids=frame_ids
                    )
                    
                    # Add to processed queue (drop oldest if full)
                    try:
                        self.processed_frames.put_nowait(frame_data)
                    except queue.Full:
                        try:
                            self.processed_frames.get_nowait()
                            self.processed_frames.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                
                # Update process FPS
                process_frame_count += 1
                current_time = time.time()
                if current_time - last_process_fps_update >= 1.0:
                    self.process_fps = process_frame_count / (current_time - last_process_fps_update)
                    process_frame_count = 0
                    last_process_fps_update = current_time
                
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                time.sleep(0.01)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess individual frame for VLA model"""
        # Resize to target dimensions
        resized = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        return normalized
    
    def cleanup(self):
        """Clean up camera and stop threads"""
        logger.info("Cleaning up vision pipeline")
        
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        logger.info("Vision pipeline cleanup completed")