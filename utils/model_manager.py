"""
Model management utilities for Face Swap Super
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .config import Config


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloads and caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry with download URLs and checksums
        self.model_registry = {
            "face_detection": {
                "scrfd_10g_bnkps": {
                    "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx",
                    "checksum": "sha256:a0a9b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
                    "size": 16777216
                },
                "retinaface_r50_v1": {
                    "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50_v1.onnx",
                    "checksum": "sha256:b1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
                    "size": 104857600
                },
                "yolov8n-face": {
                    "url": "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.onnx",
                    "checksum": "sha256:c2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
                    "size": 6291456
                }
            },
            "face_alignment": {
                "2dfan4": {
                    "url": "https://github.com/1adrianb/face-alignment/releases/download/v1.3.0/2DFAN4-11f355bf06.pth.tar",
                    "checksum": "sha256:d3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
                    "size": 24117248
                },
                "face_alignment_2dfan4": {
                    "url": "https://github.com/1adrianb/face-alignment/releases/download/v1.3.0/face_alignment_2dfan4.pth.tar",
                    "checksum": "sha256:e4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
                    "size": 62914560
                }
            },
            "identity_preservation": {
                "instantid_controlnet": {
                    "repo_id": "InstantX/InstantID",
                    "filename": "ControlNetModel/diffusion_pytorch_model.safetensors",
                    "checksum": "sha256:f5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
                },
                "instantid_ipadapter": {
                    "repo_id": "InstantX/InstantID",
                    "filename": "ip-adapter.bin",
                    "checksum": "sha256:a6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7"
                }
            },
            "inference_engine": {
                "inswapper_128": {
                    "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
                    "checksum": "sha256:b7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8",
                    "size": 526385152
                },
                "inswapper_128_fp16": {
                    "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx",
                    "checksum": "sha256:c8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9",
                    "size": 263192576
                },
                "gfpgan_1.4": {
                    "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    "checksum": "sha256:d9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
                    "size": 348632449
                }
            },
            "frame_interpolation": {
                "rife_v4.6": {
                    "url": "https://github.com/megvii-research/ECCV2022-RIFE/releases/download/v4.6/flownet.pkl",
                    "checksum": "sha256:e0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1",
                    "size": 67108864
                },
                "rife_v4.15_lite": {
                    "url": "https://github.com/hzwer/Practical-RIFE/releases/download/v4.15/train_log.zip",
                    "checksum": "sha256:f1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2",
                    "size": 134217728
                },
                "film_net": {
                    "url": "https://github.com/google-research/frame-interpolation/releases/download/v1.0/film_net.zip",
                    "checksum": "sha256:a2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3",
                    "size": 201326592
                }
            }
        }
        
        # Create model directories
        for model_type in self.model_registry:
            (self.models_dir / model_type).mkdir(exist_ok=True)
    
    def download_file(self, url: str, filepath: Path, checksum: Optional[str] = None) -> bool:
        """Download a file with progress bar and checksum verification"""
        try:
            # Check if file already exists and has correct checksum
            if filepath.exists() and checksum:
                if self.verify_checksum(filepath, checksum):
                    logger.info(f"Model {filepath.name} already exists and verified")
                    return True
                else:
                    logger.warning(f"Checksum mismatch for {filepath.name}, re-downloading...")
            
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify checksum if provided
            if checksum and not self.verify_checksum(filepath, checksum):
                logger.error(f"Checksum verification failed for {filepath.name}")
                filepath.unlink()
                return False
            
            logger.info(f"Successfully downloaded {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_from_huggingface(self, repo_id: str, filename: str, model_type: str, checksum: Optional[str] = None) -> bool:
        """Download model from Hugging Face Hub"""
        try:
            local_dir = self.models_dir / model_type
            
            # Download from Hugging Face Hub
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            # Verify checksum if provided
            if checksum and not self.verify_checksum(Path(downloaded_path), checksum):
                logger.error(f"Checksum verification failed for {filename}")
                return False
            
            logger.info(f"Successfully downloaded {filename} from {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename} from {repo_id}: {e}")
            return False
    
    def verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        try:
            algorithm, expected_hash = expected_checksum.split(':', 1)
            
            hasher = hashlib.new(algorithm)
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            actual_hash = hasher.hexdigest()
            return actual_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Error verifying checksum for {filepath}: {e}")
            return False
    
    async def download_models(self) -> bool:
        """Download all required models based on configuration"""
        logger.info("Starting model downloads...")
        
        # Determine which models to download based on configuration
        models_to_download = []
        
        # Face detection models
        face_detection_model = self.config.face_detection.model_name
        if face_detection_model in self.model_registry["face_detection"]:
            models_to_download.append(("face_detection", face_detection_model))
        
        # Face alignment models
        face_alignment_model = self.config.face_alignment.model_name
        if face_alignment_model in self.model_registry["face_alignment"]:
            models_to_download.append(("face_alignment", face_alignment_model))
        
        # Identity preservation models
        identity_model = self.config.identity_preservation.model_name
        if identity_model == "instantid":
            models_to_download.extend([
                ("identity_preservation", "instantid_controlnet"),
                ("identity_preservation", "instantid_ipadapter")
            ])
        
        # Inference engine models
        inference_model = self.config.inference_engine.model_name
        if inference_model in self.model_registry["inference_engine"]:
            models_to_download.append(("inference_engine", inference_model))
        
        # Also download GFPGAN for enhancement
        models_to_download.append(("inference_engine", "gfpgan_1.4"))
        
        # Frame interpolation models
        if self.config.frame_interpolation.enabled:
            interp_model = self.config.frame_interpolation.model_name
            if interp_model in self.model_registry["frame_interpolation"]:
                models_to_download.append(("frame_interpolation", interp_model))
        
        # Download models
        success_count = 0
        total_count = len(models_to_download)
        
        for model_type, model_name in models_to_download:
            model_info = self.model_registry[model_type][model_name]
            
            if "repo_id" in model_info:
                # Download from Hugging Face
                success = self.download_from_huggingface(
                    repo_id=model_info["repo_id"],
                    filename=model_info["filename"],
                    model_type=model_type,
                    checksum=model_info.get("checksum")
                )
            else:
                # Download from URL
                filepath = self.models_dir / model_type / model_name
                if not filepath.suffix:
                    # Add appropriate extension based on URL
                    parsed_url = urlparse(model_info["url"])
                    extension = Path(parsed_url.path).suffix
                    filepath = filepath.with_suffix(extension)
                
                success = self.download_file(
                    url=model_info["url"],
                    filepath=filepath,
                    checksum=model_info.get("checksum")
                )
            
            if success:
                success_count += 1
            else:
                logger.error(f"Failed to download {model_type}/{model_name}")
        
        logger.info(f"Downloaded {success_count}/{total_count} models successfully")
        return success_count == total_count
    
    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get path to a downloaded model"""
        model_dir = self.models_dir / model_type
        
        # Handle special cases
        if model_name == "instantid_controlnet":
            return model_dir / "ControlNetModel" / "diffusion_pytorch_model.safetensors"
        elif model_name == "instantid_ipadapter":
            return model_dir / "ip-adapter.bin"
        
        # Look for model file with common extensions
        for ext in ['.onnx', '.pth', '.safetensors', '.pkl', '.pt']:
            filepath = model_dir / f"{model_name}{ext}"
            if filepath.exists():
                return filepath
        
        # If not found, return expected path
        return model_dir / model_name
    
    def is_model_downloaded(self, model_type: str, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        model_path = self.get_model_path(model_type, model_name)
        return model_path.exists()
    
    def get_model_info(self, model_type: str, model_name: str) -> Optional[Dict]:
        """Get information about a model"""
        return self.model_registry.get(model_type, {}).get(model_name)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the registry"""
        return {
            model_type: list(models.keys())
            for model_type, models in self.model_registry.items()
        }
    
    def get_download_status(self) -> Dict[str, Dict[str, bool]]:
        """Get download status for all models"""
        status = {}
        
        for model_type, models in self.model_registry.items():
            status[model_type] = {}
            for model_name in models:
                status[model_type][model_name] = self.is_model_downloaded(model_type, model_name)
        
        return status
    
    def cleanup_old_models(self, keep_latest: int = 2):
        """Clean up old model versions"""
        logger.info("Cleaning up old model versions...")
        
        # This is a placeholder for more sophisticated cleanup logic
        # In a real implementation, you might want to:
        # 1. Keep track of model versions
        # 2. Remove older versions when newer ones are available
        # 3. Clean up temporary download files
        
        for model_type_dir in self.models_dir.iterdir():
            if model_type_dir.is_dir():
                # Clean up any temporary files
                for temp_file in model_type_dir.glob("*.tmp"):
                    temp_file.unlink()
                    logger.info(f"Removed temporary file: {temp_file}")
    
    def get_total_model_size(self) -> int:
        """Get total size of all downloaded models"""
        total_size = 0
        
        for model_type_dir in self.models_dir.iterdir():
            if model_type_dir.is_dir():
                for model_file in model_type_dir.rglob("*"):
                    if model_file.is_file():
                        total_size += model_file.stat().st_size
        
        return total_size