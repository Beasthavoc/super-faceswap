"""
Configuration management for Face Swap Super
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection module"""
    model_name: str = "scrfd_10g_bnkps"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    detection_size: int = 640
    max_faces: int = 10
    gpu_memory_fraction: float = 0.6


@dataclass
class FaceAlignmentConfig:
    """Configuration for face alignment module"""
    model_name: str = "2dfan4"
    landmark_detector: str = "face_alignment"
    align_size: int = 512
    padding_factor: float = 0.3
    use_enhancement: bool = True


@dataclass
class IdentityPreservationConfig:
    """Configuration for identity preservation module"""
    model_name: str = "instantid"
    controlnet_model: str = "InstantX/InstantID"
    adapter_model: str = "ip-adapter_instant_id_sdxl"
    strength: float = 0.8
    guidance_scale: float = 7.5
    num_inference_steps: int = 20


@dataclass
class InferenceEngineConfig:
    """Configuration for real-time inference engine"""
    model_name: str = "inswapper_128"
    face_swapper: str = "facefusion"
    batch_size: int = 1
    num_threads: int = 4
    optimization_level: int = 3
    precision: str = "fp16"
    cache_size: int = 100


@dataclass
class FrameInterpolationConfig:
    """Configuration for frame interpolation"""
    enabled: bool = True
    model_name: str = "rife_v4.6"
    interpolation_factor: int = 2
    enhance_quality: bool = True
    use_film_fallback: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration"""
    device: str = "auto"  # auto, cuda, cpu, mps
    gpu_ids: list = field(default_factory=lambda: [0])
    num_workers: int = 4
    memory_limit: str = "8GB"
    temp_dir: str = "/tmp/face_swap_super"
    log_level: str = "INFO"
    enable_profiling: bool = False


@dataclass
class SecurityConfig:
    """Security and safety configuration"""
    enable_nsfw_filter: bool = True
    enable_deepfake_detection: bool = True
    watermark_enabled: bool = True
    content_moderation: bool = True
    rate_limiting: bool = True
    max_concurrent_requests: int = 10


@dataclass
class WebConfig:
    """Web interface configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    websocket_enabled: bool = True
    max_file_size: str = "100MB"
    session_timeout: int = 3600


class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize with defaults
        self.face_detection = FaceDetectionConfig()
        self.face_alignment = FaceAlignmentConfig()
        self.identity_preservation = IdentityPreservationConfig()
        self.inference_engine = InferenceEngineConfig()
        self.frame_interpolation = FrameInterpolationConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.web = WebConfig()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
        
        # Create necessary directories
        self.create_directories()
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration sections
            if 'face_detection' in config_data:
                self._update_config(self.face_detection, config_data['face_detection'])
            
            if 'face_alignment' in config_data:
                self._update_config(self.face_alignment, config_data['face_alignment'])
            
            if 'identity_preservation' in config_data:
                self._update_config(self.identity_preservation, config_data['identity_preservation'])
            
            if 'inference_engine' in config_data:
                self._update_config(self.inference_engine, config_data['inference_engine'])
            
            if 'frame_interpolation' in config_data:
                self._update_config(self.frame_interpolation, config_data['frame_interpolation'])
            
            if 'system' in config_data:
                self._update_config(self.system, config_data['system'])
            
            if 'security' in config_data:
                self._update_config(self.security, config_data['security'])
            
            if 'web' in config_data:
                self._update_config(self.web, config_data['web'])
                
        except Exception as e:
            print(f"Error loading config file {config_file}: {e}")
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'FACESWAP_DEVICE': ('system', 'device'),
            'FACESWAP_GPU_IDS': ('system', 'gpu_ids'),
            'FACESWAP_LOG_LEVEL': ('system', 'log_level'),
            'FACESWAP_HOST': ('web', 'host'),
            'FACESWAP_PORT': ('web', 'port'),
            'FACESWAP_ENABLE_NSFW_FILTER': ('security', 'enable_nsfw_filter'),
            'FACESWAP_WATERMARK_ENABLED': ('security', 'watermark_enabled'),
            'FACESWAP_FRAME_INTERPOLATION': ('frame_interpolation', 'enabled'),
            'FACESWAP_INTERPOLATION_FACTOR': ('frame_interpolation', 'interpolation_factor'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                
                # Convert string values to appropriate types
                if hasattr(section_obj, key):
                    current_value = getattr(section_obj, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, list):
                        value = [int(x.strip()) for x in value.split(',')]
                    
                    setattr(section_obj, key, value)
    
    def _update_config(self, config_obj, config_dict):
        """Update configuration object with dictionary values"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "models",
            "models/face_detection",
            "models/face_alignment", 
            "models/identity_preservation",
            "models/inference_engine",
            "models/frame_interpolation",
            "assets",
            "assets/samples",
            "assets/temp",
            "outputs",
            "logs",
            self.system.temp_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'face_detection': self.face_detection.__dict__,
            'face_alignment': self.face_alignment.__dict__,
            'identity_preservation': self.identity_preservation.__dict__,
            'inference_engine': self.inference_engine.__dict__,
            'frame_interpolation': self.frame_interpolation.__dict__,
            'system': self.system.__dict__,
            'security': self.security.__dict__,
            'web': self.web.__dict__,
        }
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML file"""
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def get_model_path(self, model_type: str, model_name: str) -> str:
        """Get path to a model file"""
        return os.path.join("models", model_type, model_name)
    
    def get_asset_path(self, asset_name: str) -> str:
        """Get path to an asset file"""
        return os.path.join("assets", asset_name)
    
    def get_output_path(self, output_name: str) -> str:
        """Get path to an output file"""
        return os.path.join("outputs", output_name)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate face detection config
        if self.face_detection.confidence_threshold < 0 or self.face_detection.confidence_threshold > 1:
            errors.append("Face detection confidence threshold must be between 0 and 1")
        
        # Validate system config
        if self.system.device not in ['auto', 'cuda', 'cpu', 'mps']:
            errors.append("Invalid device type")
        
        # Validate web config
        if self.web.port < 1 or self.web.port > 65535:
            errors.append("Invalid port number")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Default configuration as YAML
DEFAULT_CONFIG = """
face_detection:
  model_name: "scrfd_10g_bnkps"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  detection_size: 640
  max_faces: 10
  gpu_memory_fraction: 0.6

face_alignment:
  model_name: "2dfan4"
  landmark_detector: "face_alignment"
  align_size: 512
  padding_factor: 0.3
  use_enhancement: true

identity_preservation:
  model_name: "instantid"
  controlnet_model: "InstantX/InstantID"
  adapter_model: "ip-adapter_instant_id_sdxl"
  strength: 0.8
  guidance_scale: 7.5
  num_inference_steps: 20

inference_engine:
  model_name: "inswapper_128"
  face_swapper: "facefusion"
  batch_size: 1
  num_threads: 4
  optimization_level: 3
  precision: "fp16"
  cache_size: 100

frame_interpolation:
  enabled: true
  model_name: "rife_v4.6"
  interpolation_factor: 2
  enhance_quality: true
  use_film_fallback: true

system:
  device: "auto"
  gpu_ids: [0]
  num_workers: 4
  memory_limit: "8GB"
  temp_dir: "/tmp/face_swap_super"
  log_level: "INFO"
  enable_profiling: false

security:
  enable_nsfw_filter: true
  enable_deepfake_detection: true
  watermark_enabled: true
  content_moderation: true
  rate_limiting: true
  max_concurrent_requests: 10

web:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  websocket_enabled: true
  max_file_size: "100MB"
  session_timeout: 3600
"""


def create_default_config(config_file: str = "config.yaml"):
    """Create a default configuration file"""
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write(DEFAULT_CONFIG)
        print(f"Created default configuration file: {config_file}")
    else:
        print(f"Configuration file already exists: {config_file}")


if __name__ == "__main__":
    # Create default config if run directly
    create_default_config()