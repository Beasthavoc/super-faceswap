"""
Device management utilities for Face Swap Super
"""

import torch
import platform
import logging
from typing import Union, List, Optional


logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and allocation for the application"""
    
    def __init__(self):
        self.device = None
        self.available_devices = []
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available devices"""
        self.available_devices = []
        
        # Always have CPU available
        self.available_devices.append("cpu")
        
        # Check for CUDA
        if self.cuda_available:
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                device_name = torch.cuda.get_device_name(i)
                self.available_devices.append(f"cuda:{i}")
                logger.info(f"CUDA device {i}: {device_name}")
        
        # Check for MPS (Apple Silicon)
        if self.mps_available:
            self.available_devices.append("mps")
            logger.info("MPS device available")
        
        logger.info(f"Available devices: {self.available_devices}")
    
    def get_device(self, device_preference: str = "auto") -> torch.device:
        """Get the best available device based on preference"""
        
        if device_preference == "auto":
            # Auto-select best device
            if self.cuda_available:
                device = torch.device("cuda:0")
            elif self.mps_available:
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            # Use specified device
            device = torch.device(device_preference)
        
        self.device = device
        logger.info(f"Selected device: {device}")
        return device
    
    def get_device_info(self) -> dict:
        """Get detailed information about the current device"""
        if self.device is None:
            self.get_device()
        
        info = {
            "device": str(self.device),
            "type": self.device.type,
            "available_devices": self.available_devices,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
        }
        
        if self.device.type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "current_gpu": self.device.index,
                "gpu_name": torch.cuda.get_device_name(self.device),
                "gpu_memory_total": torch.cuda.get_device_properties(self.device).total_memory,
                "gpu_memory_reserved": torch.cuda.memory_reserved(self.device),
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device),
            })
        
        return info
    
    def set_memory_fraction(self, fraction: float = 0.8):
        """Set memory fraction for GPU usage"""
        if self.device and self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)
            logger.info(f"Set GPU memory fraction to {fraction}")
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
    
    def get_memory_info(self) -> dict:
        """Get memory information for the current device"""
        if self.device and self.device.type == "cuda":
            return {
                "total": torch.cuda.get_device_properties(self.device).total_memory,
                "reserved": torch.cuda.memory_reserved(self.device),
                "allocated": torch.cuda.memory_allocated(self.device),
                "free": torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_reserved(self.device),
            }
        else:
            return {"message": "Memory info only available for CUDA devices"}
    
    def optimize_for_inference(self):
        """Optimize device settings for inference"""
        if self.device and self.device.type == "cuda":
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set memory management
            torch.cuda.empty_cache()
            
            logger.info("Optimized CUDA settings for inference")
    
    def can_use_mixed_precision(self) -> bool:
        """Check if mixed precision can be used"""
        if self.device and self.device.type == "cuda":
            # Check if GPU supports mixed precision
            return torch.cuda.get_device_capability(self.device)[0] >= 7
        return False
    
    def get_optimal_batch_size(self, model_size_mb: int = 500) -> int:
        """Get optimal batch size based on available memory"""
        if self.device and self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = total_memory - torch.cuda.memory_reserved(self.device)
            
            # Rough estimation: leave 20% buffer for other operations
            usable_memory = available_memory * 0.8
            
            # Estimate batch size (very rough approximation)
            estimated_batch_size = max(1, int(usable_memory / (model_size_mb * 1024 * 1024)))
            
            return min(estimated_batch_size, 32)  # Cap at 32 for safety
        else:
            return 1  # Conservative batch size for CPU
    
    def is_device_available(self, device_name: str) -> bool:
        """Check if a specific device is available"""
        return device_name in self.available_devices
    
    def get_compute_capability(self) -> Optional[tuple]:
        """Get CUDA compute capability"""
        if self.device and self.device.type == "cuda":
            return torch.cuda.get_device_capability(self.device)
        return None