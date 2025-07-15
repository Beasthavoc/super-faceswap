#!/usr/bin/env python3
"""
Face Swap Super - Main Application
A comprehensive real-time face swapping system combining the best technologies
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import gradio as gr

# Import our modules
from face_detection.face_detector import FaceDetector
from face_alignment.face_aligner import FaceAligner
from identity_preservation.identity_preserver import IdentityPreserver
from inference_engine.real_time_engine import RealTimeEngine
from frame_interpolation.frame_interpolator import FrameInterpolator
from utils.config import Config
from utils.logger import setup_logging
from utils.device_manager import DeviceManager
from utils.model_manager import ModelManager

# Setup logging
logger = setup_logging()

class FaceSwapSuper:
    """Main application class that orchestrates all components"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(config)
        
        # Initialize components
        self.face_detector = None
        self.face_aligner = None
        self.identity_preserver = None
        self.inference_engine = None
        self.frame_interpolator = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Face Swap Super...")
        
        # Download and load models
        await self.model_manager.download_models()
        
        # Initialize components
        self.face_detector = FaceDetector(self.config)
        self.face_aligner = FaceAligner(self.config)
        self.identity_preserver = IdentityPreserver(self.config)
        self.inference_engine = RealTimeEngine(self.config)
        self.frame_interpolator = FrameInterpolator(self.config)
        
        # Load models
        await self.face_detector.load_models()
        await self.face_aligner.load_models()
        await self.identity_preserver.load_models()
        await self.inference_engine.load_models()
        await self.frame_interpolator.load_models()
        
        logger.info("Face Swap Super initialized successfully!")
        
    async def process_frame(self, frame_data: bytes) -> bytes:
        """Process a single frame through the complete pipeline"""
        try:
            # 1. Face Detection
            faces = await self.face_detector.detect_faces(frame_data)
            
            # 2. Face Alignment
            aligned_faces = await self.face_aligner.align_faces(faces)
            
            # 3. Identity Preservation
            preserved_faces = await self.identity_preserver.preserve_identity(aligned_faces)
            
            # 4. Real-time Inference
            swapped_frame = await self.inference_engine.process_frame(preserved_faces)
            
            # 5. Frame Interpolation (if needed)
            enhanced_frame = await self.frame_interpolator.interpolate(swapped_frame)
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame_data  # Return original frame on error
    
    async def process_video(self, input_path: str, output_path: str, source_face: str) -> str:
        """Process a complete video file"""
        logger.info(f"Processing video: {input_path}")
        
        try:
            # Use inference engine for video processing
            result = await self.inference_engine.process_video(
                input_path, output_path, source_face
            )
            
            # Apply frame interpolation if configured
            if self.config.frame_interpolation.enabled:
                result = await self.frame_interpolator.process_video(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            raise
    
    def create_gradio_interface(self):
        """Create Gradio web interface"""
        
        def process_video_gradio(input_video, source_image):
            """Gradio wrapper for video processing"""
            if not input_video or not source_image:
                return None, "Please provide both input video and source image"
            
            try:
                # Create temporary output path
                output_path = "outputs/gradio_output.mp4"
                
                # Process video
                result = asyncio.run(
                    self.process_video(input_video, output_path, source_image)
                )
                
                return result, "Video processed successfully!"
                
            except Exception as e:
                return None, f"Error processing video: {str(e)}"
        
        # Create interface
        interface = gr.Interface(
            fn=process_video_gradio,
            inputs=[
                gr.Video(label="Input Video"),
                gr.Image(label="Source Face", type="filepath")
            ],
            outputs=[
                gr.Video(label="Output Video"),
                gr.Textbox(label="Status")
            ],
            title="Face Swap Super",
            description="High-performance real-time face swapping system",
            examples=[
                ["assets/sample_video.mp4", "assets/sample_face.jpg"]
            ]
        )
        
        return interface


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Face Swap Super")
    parser.add_argument("--mode", choices=["gradio", "cli"], default="gradio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="Input video file (CLI mode)")
    parser.add_argument("--output", help="Output video file (CLI mode)")
    parser.add_argument("--source", help="Source face image (CLI mode)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = Config(args.config)
    
    # Create application
    app = FaceSwapSuper(config)
    
    if args.mode == "gradio":
        # Run as Gradio interface
        async def run_gradio():
            await app.initialize()
            interface = app.create_gradio_interface()
            interface.launch(
                server_name=args.host,
                server_port=args.port,
                share=False
            )
        
        asyncio.run(run_gradio())
        
    elif args.mode == "cli":
        # Run as CLI application
        if not all([args.input, args.output, args.source]):
            print("CLI mode requires --input, --output, and --source arguments")
            sys.exit(1)
        
        async def run_cli():
            await app.initialize()
            result = await app.process_video(args.input, args.output, args.source)
            print(f"Video processed successfully: {result}")
        
        asyncio.run(run_cli())


if __name__ == "__main__":
    main()