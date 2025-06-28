"""
Whisper Model Handler
Handles loading Whisper models and performing transcription with real-time pipeline
"""
import whisper
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, List
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperModel:
    """Wrapper class for OpenAI Whisper model with real-time pipeline support"""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper model
        
        Args:
            model_size: Size of the model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu'). If None, automatically selects.
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        
        # Pipeline settings
        self.segment_duration = 20  # seconds
        self.overlap_ratio = 0.5  # 50% overlap
        self.sample_rate = 16000
        
        logger.info(f"Initializing Whisper model: {model_size} on {self.device}")
        
    def load_model(self) -> bool:
        """Load the Whisper model - simple like main.py"""
        try:
            logger.info(f"Loading model '{self.model_size}' on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.model_loaded = True
            logger.info(f"✓ Model '{self.model_size}' loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            return False
            
    def change_model(self, new_model_size: str) -> bool:
        """Change to a different model size"""
        if new_model_size == self.model_size and self.model_loaded:
            return True
            
        # Unload current model
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            
        # Load new model
        self.model_size = new_model_size
        return self.load_model()
        
    def transcribe_audio(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr'). None for auto-detection.
            task: 'transcribe' or 'translate' (to English)
            **kwargs: Additional arguments for whisper.transcribe()
            
        Returns:
            Dictionary with transcription results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=(self.device == "cuda"),
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"✗ Error during transcription: {e}")
            raise
            
    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio from numpy array
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language: Language code (e.g., 'en', 'es', 'fr'). None for auto-detection.
            task: 'transcribe' or 'translate' (to English)
            **kwargs: Additional arguments for whisper.transcribe()
            
        Returns:
            Dictionary with transcription results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Ensure audio is float32 and normalized
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            
        # Normalize if needed
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
            
        try:
            result = self.model.transcribe(
                audio_array,
                language=language,
                task=task,
                fp16=(self.device == "cuda"),
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"✗ Error during transcription: {e}")
            raise
            
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio segment (for real-time pipeline)
        
        Args:
            audio_segment: Audio segment as numpy array
            language: Language code
            task: 'transcribe' or 'translate'
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with transcription results
        """
        return self.transcribe_array(
            audio_segment, 
            self.sample_rate, 
            language, 
            task, 
            **kwargs
        )
        
    def merge_transcripts(self, transcripts: List[Dict[str, Any]]) -> str:
        """
        Merge overlapping transcripts into a single chronological transcript
        
        Args:
            transcripts: List of transcript dictionaries
            
        Returns:
            Merged transcript text
        """
        if not transcripts:
            return ""
            
        # Simple merge - in a real implementation, you'd want more sophisticated
        # overlap detection and merging
        merged_text = ""
        for transcript in transcripts:
            text = transcript.get('text', '').strip()
            if text:
                if merged_text and not merged_text.endswith(' '):
                    merged_text += ' '
                merged_text += text
                
        return merged_text.strip()
        
    def get_available_languages(self) -> list:
        """Get list of available languages"""
        return list(whisper.tokenizer.LANGUAGES.values())
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"status": "not_loaded", "model_size": self.model_size}
            
        return {
            "status": "loaded",
            "model_size": self.model_size,
            "device": str(self.device),
            "n_mels": self.model.dims.n_mels,
            "n_vocab": self.model.dims.n_vocab,
            "n_audio_ctx": self.model.dims.n_audio_ctx,
            "n_audio_state": self.model.dims.n_audio_state,
            "n_audio_head": self.model.dims.n_audio_head,
            "n_audio_layer": self.model.dims.n_audio_layer,
        }
        
    def get_available_models(self) -> List[str]:
        """Get list of available model sizes"""
        return ["tiny", "base", "small", "medium", "large"] 