"""
Configuration module for SoumyaGPT.
Handles API keys, model paths, and system settings.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json


class Config:
    """Configuration class for SoumyaGPT."""

    def __init__(self):
        # Base paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.voice_samples_dir = self.data_dir / "voice_samples"
        self.models_dir = self.data_dir / "models"
        
        # Ensure directories exist
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # API Configuration
        self.gemini_api_key = ("AIzaSyCor2N40sQNVjEp9ilQ6G1xBIyKgh1BbtA")
        
        # Model configurations
        self.whisper_model = "base"  # faster-whisper model size
        self.whisper_device = "auto"  # auto, cpu, cuda
        self.whisper_compute_type = "float16"  # int8, float16, float32
        
        # TTS Configuration
        self.tts_engine = "edge-tts"  # "edge-tts" or "pyttsx3"
        self.edge_voice = "en-US-AriaNeural"  # Default Edge TTS voice
        self.pyttsx3_rate = 150  # Speech rate for pyttsx3
        self.pyttsx3_volume = 0.9  # Volume for pyttsx3
        
        # Voice cloning configuration
        self.tts_model_name = "xtts"  # Use XTTS for voice cloning
        self.voice_clone_model = "xtts"  # XTTS voice cloning
        self.reference_audio_path = self.voice_samples_dir / "obama-original.wav"  # Use Obama voice
        
        # Audio settings
        self.sample_rate = 22050
        self.default_duration = 5.0  # seconds
        self.audio_chunk_size = 1024
        
        # System settings
        self.max_response_tokens = 1000
        self.temperature = 0.7
        self.verbose = False
        
        # Load custom config if exists
        self._load_custom_config()
    
    def _get_env_var(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with fallback."""
        value = os.getenv(var_name, default)
        if not value and var_name == "GEMINI_API_KEY":
            print(f"⚠️  Warning: {var_name} not found in environment variables.")
            print("   Please set it using: set GEMINI_API_KEY=your_key_here")
        return value
    
    def _load_custom_config(self):
        """Load custom configuration from config.json if it exists."""
        config_file = self.project_root / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                
                # Update configuration with custom values
                for key, value in custom_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not load config.json: {e}")
    
    def save_custom_config(self, config_dict: Dict[str, Any]):
        """Save custom configuration to config.json."""
        config_file = self.project_root / "config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def validate(self) -> bool:
        """Validate configuration and check for required components."""
        issues = []
        
        # Check API key
        if not self.gemini_api_key:
            issues.append("GEMINI_API_KEY not set")
        
        # Check if reference audio exists for voice cloning
        if not self.reference_audio_path.exists():
            issues.append(f"Reference audio not found: {self.reference_audio_path}")
            print(f"ℹ️  Place your voice sample at: {self.reference_audio_path}")
        
        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True
    
    def get_whisper_config(self) -> Dict[str, Any]:
        """Get Whisper configuration dictionary."""
        return {
            "model_size_or_path": self.whisper_model,
            "device": self.whisper_device,
            "compute_type": self.whisper_compute_type,
        }
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS configuration dictionary."""
        return {
            "engine": self.tts_engine,
            "edge_voice": self.edge_voice,
            "pyttsx3_rate": self.pyttsx3_rate,
            "pyttsx3_volume": self.pyttsx3_volume,
            "reference_audio": str(self.reference_audio_path),
        }
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini API configuration."""
        return {
            "api_key": self.gemini_api_key,
            "model_name": "gemini-2.0-flash-exp",  # Latest Gemini 2.0 Flash model
            "generation_config": {
                "temperature": self.temperature,
                "max_output_tokens": self.max_response_tokens,
            }
        }
    
    def print_config(self):
        """Print current configuration."""
        print("🔧 SoumyaGPT Configuration:")
        print(f"   Project Root: {self.project_root}")
        print(f"   Voice Samples: {self.voice_samples_dir}")
        print(f"   Whisper Model: {self.whisper_model} ({self.whisper_device})")
        print(f"   TTS Engine: {self.tts_engine}")
        print(f"   Edge Voice: {self.edge_voice}")
        print(f"   Reference Audio: {self.reference_audio_path}")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Gemini API: {'✅ Configured' if self.gemini_api_key else '❌ Missing'}")


# Global configuration instance
config = Config()