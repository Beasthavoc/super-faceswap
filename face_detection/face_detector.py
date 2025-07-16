"""
Face Detection Module for Face Swap Super
Using InsightFace/SCRFD technology for high-performance face detection
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import asyncio
import time

from ..utils.config import Config
from ..utils.device_manager import DeviceManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Face:
    """Face detection result"""
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray, confidence: float, embedding: Optional[np.ndarray] = None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.landmarks = landmarks  # 5 point landmarks
        self.confidence = confidence
        self.embedding = embedding
        
        # Calculate face area and center
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Calculate face angle
        self.angle = self._calculate_angle()
    
    def _calculate_angle(self) -> float:
        """Calculate face rotation angle from landmarks"""
        if self.landmarks is not None and len(self.landmarks) >= 2:
            # Use eye landmarks to calculate angle
            left_eye = self.landmarks[0]
            right_eye = self.landmarks[1]
            
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            
            return np.arctan2(delta_y, delta_x) * 180 / np.pi
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert face to dictionary"""
        return {
            'bbox': self.bbox.tolist(),
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None,
            'confidence': float(self.confidence),
            'area': float(self.area),
            'center': self.center,
            'angle': float(self.angle)
        }


class FaceDetector:
    """High-performance face detector using SCRFD"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device_manager = DeviceManager()
        self.model = None
        self.session = None
        self.input_shape = (640, 640)
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        
        # Initialize ONNX Runtime providers
        self.providers = self._get_providers()
        
    def _get_providers(self) -> List[str]:
        """Get optimal ONNX Runtime providers"""
        providers = []
        
        device = self.device_manager.get_device()
        
        if device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        elif device.type == 'cpu':
            providers.append('CPUExecutionProvider')
        
        return providers
    
    async def load_models(self):
        """Load face detection model"""
        logger.info("Loading face detection models...")
        
        try:
            # Get model path
            model_path = self.config.get_model_path(
                "face_detection", 
                f"{self.config.face_detection.model_name}.onnx"
            )
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set number of threads for CPU
            if 'CPUExecutionProvider' in self.providers:
                session_options.intra_op_num_threads = self.config.inference_engine.num_threads
            
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=self.providers
            )
            
            # Get input shape from model
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:
                self.input_shape = (input_shape[2], input_shape[3])  # (height, width)
            
            logger.info(f"Face detection model loaded successfully")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Providers: {self.providers}")
            
        except Exception as e:
            logger.error(f"Error loading face detection model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for face detection"""
        # Get original image dimensions
        height, width = image.shape[:2]
        
        # Calculate scale factor
        scale = min(self.input_shape[0] / height, self.input_shape[1] / width)
        
        # Resize image
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded_image = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        padded_image[:new_height, :new_width] = resized_image
        
        # Convert to float and normalize
        padded_image = padded_image.astype(np.float32)
        padded_image = (padded_image - 127.5) / 128.0
        
        # Change to CHW format
        padded_image = np.transpose(padded_image, (2, 0, 1))
        padded_image = np.expand_dims(padded_image, axis=0)
        
        return padded_image, scale
    
    def postprocess_detections(self, outputs: List[np.ndarray], scale: float, original_shape: Tuple[int, int]) -> List[Face]:
        """Post-process detection outputs"""
        faces = []
        
        try:
            # Parse outputs (format depends on model)
            if len(outputs) >= 3:
                # SCRFD format: [boxes, scores, landmarks]
                boxes = outputs[0]
                scores = outputs[1]
                landmarks = outputs[2] if len(outputs) > 2 else None
                
                # Process detections
                for i in range(boxes.shape[0]):
                    confidence = scores[i]
                    
                    # Filter by confidence threshold
                    if confidence < self.config.face_detection.confidence_threshold:
                        continue
                    
                    # Scale box coordinates back to original image
                    box = boxes[i] / scale
                    
                    # Ensure box is within image bounds
                    box[0] = max(0, box[0])
                    box[1] = max(0, box[1])
                    box[2] = min(original_shape[1], box[2])
                    box[3] = min(original_shape[0], box[3])
                    
                    # Process landmarks if available
                    face_landmarks = None
                    if landmarks is not None:
                        face_landmarks = landmarks[i].reshape(-1, 2) / scale
                    
                    # Create face object
                    face = Face(
                        bbox=box,
                        landmarks=face_landmarks,
                        confidence=confidence
                    )
                    
                    faces.append(face)
            
            # Apply NMS (Non-Maximum Suppression)
            faces = self._apply_nms(faces)
            
            # Sort by confidence and limit max faces
            faces = sorted(faces, key=lambda x: x.confidence, reverse=True)
            faces = faces[:self.config.face_detection.max_faces]
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
        
        return faces
    
    def _apply_nms(self, faces: List[Face]) -> List[Face]:
        """Apply Non-Maximum Suppression"""
        if not faces:
            return faces
        
        # Convert to format expected by OpenCV NMS
        boxes = np.array([face.bbox for face in faces])
        scores = np.array([face.confidence for face in faces])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config.face_detection.confidence_threshold,
            self.config.face_detection.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        
        return []
    
    async def detect_faces(self, image: np.ndarray) -> List[Face]:
        """Detect faces in image"""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image, scale = self.preprocess_image(image)
            
            # Get input name
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            outputs = self.session.run(None, {input_name: processed_image})
            
            # Postprocess results
            faces = self.postprocess_detections(outputs, scale, image.shape[:2])
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(faces)
            
            logger.debug(f"Detected {len(faces)} faces in {inference_time:.3f}s")
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def detect_faces_batch(self, images: List[np.ndarray]) -> List[List[Face]]:
        """Detect faces in batch of images"""
        # For now, process sequentially
        # TODO: Implement true batch processing
        results = []
        
        for image in images:
            faces = asyncio.run(self.detect_faces(image))
            results.append(faces)
        
        return results
    
    def get_largest_face(self, faces: List[Face]) -> Optional[Face]:
        """Get the largest face from detection results"""
        if not faces:
            return None
        
        return max(faces, key=lambda face: face.area)
    
    def filter_faces_by_size(self, faces: List[Face], min_area: int = 1000) -> List[Face]:
        """Filter faces by minimum area"""
        return [face for face in faces if face.area >= min_area]
    
    def get_face_embeddings(self, faces: List[Face], image: np.ndarray) -> List[Face]:
        """Get face embeddings (placeholder for now)"""
        # TODO: Implement face embedding extraction
        # This would typically use a separate face recognition model
        return faces
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {"message": "No inference statistics available"}
        
        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        
        return {
            "total_inferences": len(self.inference_times),
            "total_detections": self.total_detections,
            "avg_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "avg_faces_per_inference": self.total_detections / len(self.inference_times)
        }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.inference_times = []
        self.total_detections = 0
    
    def visualize_detections(self, image: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Visualize face detections on image"""
        vis_image = image.copy()
        
        for face in faces:
            # Draw bounding box
            pt1 = (int(face.bbox[0]), int(face.bbox[1]))
            pt2 = (int(face.bbox[2]), int(face.bbox[3]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(
                vis_image,
                f"{face.confidence:.2f}",
                (pt1[0], pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw landmarks if available
            if face.landmarks is not None:
                for landmark in face.landmarks:
                    cv2.circle(vis_image, (int(landmark[0]), int(landmark[1])), 2, (255, 0, 0), -1)
        
        return vis_image
    
    def __del__(self):
        """Cleanup resources"""
        if self.session:
            self.session = None