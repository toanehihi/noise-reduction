import os
import sys
import logging
import numpy as np
import soundfile as sf
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import DTLN model
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from model import DTLN_model

logger = logging.getLogger(__name__)


class NoiseReductionService:
    """
    Service for performing noise reduction using DTLN model.
    
    This service can be used standalone or integrated into any application.
    
    Example usage:
        # Initialize service
        service = NoiseReductionService(model_path='path/to/model.h5')
        
        # Process audio file
        service.process_audio('input.wav', 'output.wav')
        
        # Or process audio data directly
        denoised = service.process_audio_data(audio_array)
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to cache model in memory"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path, sample_rate=16000):
        """
        Initialize the noise reduction service
        
        Args:
            model_path: Path to trained DTLN model weights (.h5 file)
            sample_rate: Expected sample rate (default: 16000 Hz)
        """
        if self._initialized:
            return
            
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self.dtln = None
        
        # Load model
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load DTLN model from weights file"""
        try:
            logger.info(f"Loading DTLN model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize DTLN model
            self.dtln = DTLN_model()
            
            # Check if model uses normalization (based on filename)
            norm_stft = '_norm_' in os.path.basename(self.model_path)
            
            # Build model architecture
            self.dtln.build_DTLN_model(norm_stft=norm_stft)
            
            # Load weights
            self.dtln.model.load_weights(self.model_path)
            self.model = self.dtln.model
            
            logger.info("Model loaded successfully")
            logger.info(f"Model uses normalization: {norm_stft}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_audio(self, input_path, output_path):
        """
        Process audio file to remove noise
        
        Args:
            input_path: Path to input noisy audio file
            output_path: Path to save denoised audio file
            
        Returns:
            output_path: Path to the denoised audio file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If audio format is invalid
        """
        try:
            logger.info(f"Processing audio file: {input_path}")
            
            # Validate input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read audio file
            audio, fs = sf.read(input_path)
            
            # Validate sample rate
            if fs != self.sample_rate:
                logger.warning(
                    f"Audio sample rate ({fs} Hz) doesn't match expected "
                    f"rate ({self.sample_rate} Hz). Quality may be affected."
                )
            
            # Ensure mono audio
            if audio.ndim > 1:
                logger.info("Converting stereo to mono")
                audio = np.mean(audio, axis=1)
            
            # Process audio
            denoised_audio = self.process_audio_data(audio)
            
            # Save denoised audio
            sf.write(output_path, denoised_audio, self.sample_rate)
            
            logger.info(f"Denoised audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise
    
    def process_audio_data(self, audio):
        """
        Process audio data using DTLN model
        
        Args:
            audio: numpy array of audio samples
            
        Returns:
            denoised_audio: numpy array of denoised audio samples
            
        Raises:
            ValueError: If audio data format is invalid
        """
        try:
            # Validate input
            if not isinstance(audio, np.ndarray):
                raise ValueError("Audio must be a numpy array")
            
            # Ensure audio is float32
            audio = audio.astype('float32')
            
            # Ensure 1D array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Add batch dimension
            audio_batch = np.expand_dims(audio, axis=0)
            
            # Run inference
            logger.info(f"Running inference on audio shape: {audio_batch.shape}")
            denoised_audio = self.model.predict(audio_batch, verbose=0)
            
            # Remove batch dimension
            denoised_audio = np.squeeze(denoised_audio)
            
            # Clip to valid range [-1, 1]
            denoised_audio = np.clip(denoised_audio, -1.0, 1.0)
            
            logger.info(f"Inference complete. Output shape: {denoised_audio.shape}")
            return denoised_audio
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise
    
    def is_ready(self):
        """Check if service is ready to process audio"""
        return self.model is not None
    
    def get_info(self):
        """Get information about the service"""
        if not self.is_ready():
            return {
                "status": "not_ready",
                "error": "Model not loaded"
            }
        
        return {
            "status": "ready",
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape)
        }
