"""
Speech-to-Text module using faster-whisper for GPU-optimized transcription.
Supports both microphone input and audio file transcription.
"""

import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Optional, Union, Tuple
import warnings

# Suppress warnings from faster-whisper
warnings.filterwarnings("ignore", module="faster_whisper")

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster-whisper not installed. Run: pip install faster-whisper")
    raise

from .config import config


class SpeechToText:
    """Speech-to-Text processor using faster-whisper."""
    
    def __init__(self):
        self.model = None
        self.sample_rate = config.sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            whisper_config = config.get_whisper_config()
            
            print(f"🔄 Loading Whisper model: {whisper_config['model_size_or_path']}")
            print(f"   Device: {whisper_config['device']}")
            print(f"   Compute type: {whisper_config['compute_type']}")
            
            self.model = WhisperModel(
                whisper_config["model_size_or_path"],
                device=whisper_config["device"],
                compute_type=whisper_config["compute_type"]
            )
            
            print("✅ Whisper model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            print("💡 Tip: Try using a smaller model (tiny, base) or CPU device")
            raise
    
    def record_audio(self, duration: float = None) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds. Uses config default if None.
            
        Returns:
            numpy array of audio data
        """
        if duration is None:
            duration = config.default_duration
        
        try:
            print(f"🎤 Recording for {duration} seconds... (speak now!)")
            
            # Record audio
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            print("✅ Recording completed")
            return audio.flatten()
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            raise
    
    def load_audio_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            numpy array of audio data
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            print(f"📁 Loading audio file: {file_path}")
            
            # Load audio file
            audio, sr = sf.read(str(file_path))
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if necessary
            if sr != self.sample_rate:
                print(f"🔄 Resampling from {sr} Hz to {self.sample_rate} Hz")
                # Simple resampling (for better quality, use librosa.resample)
                import scipy.signal
                audio = scipy.signal.resample(
                    audio, 
                    int(len(audio) * self.sample_rate / sr)
                )
            
            print("✅ Audio file loaded successfully")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            raise
    
    def transcribe_audio(self, audio: np.ndarray, language: str = "en") -> Tuple[str, float]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array
            language: Language code (e.g., "en", "es", "fr")
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        if self.model is None:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            print("🧠 Transcribing audio...")
            
            # Save audio to temporary file for faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                temp_path = temp_file.name
            
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                temp_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Combine all segments
            transcription = ""
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                transcription += segment.text
                total_confidence += segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0
                segment_count += 1
            
            # Calculate average confidence
            confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            # Clean up transcription
            transcription = transcription.strip()
            
            if not transcription:
                print("⚠️  No speech detected in audio")
                return "", 0.0
            
            print(f"✅ Transcription completed (confidence: {confidence:.2f})")
            print(f"📝 Text: {transcription}")
            
            return transcription, confidence
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            raise
    
    def transcribe_microphone(
        self, 
        duration: float = None, 
        language: str = "en"
    ) -> Tuple[str, float]:
        """
        Record from microphone and transcribe.
        
        Args:
            duration: Recording duration in seconds
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Record audio
            audio = self.record_audio(duration)
            
            # Transcribe
            return self.transcribe_audio(audio, language)
            
        except Exception as e:
            print(f"❌ Error in microphone transcription: {e}")
            raise
    
    def transcribe_file(
        self, 
        file_path: Union[str, Path], 
        language: str = "en"
    ) -> Tuple[str, float]:
        """
        Load audio file and transcribe.
        
        Args:
            file_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Load audio file
            audio = self.load_audio_file(file_path)
            
            # Transcribe
            return self.transcribe_audio(audio, language)
            
        except Exception as e:
            print(f"❌ Error in file transcription: {e}")
            raise
    
    def is_audio_valid(self, audio: np.ndarray, min_duration: float = 0.5) -> bool:
        """
        Check if audio data is valid and long enough.
        
        Args:
            audio: Audio data
            min_duration: Minimum duration in seconds
            
        Returns:
            True if audio is valid
        """
        if audio is None or len(audio) == 0:
            return False
        
        duration = len(audio) / self.sample_rate
        if duration < min_duration:
            print(f"⚠️  Audio too short: {duration:.2f}s (minimum: {min_duration}s)")
            return False
        
        # Check if audio has sufficient energy (not just silence)
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < 0.001:  # Very quiet threshold
            print("⚠️  Audio appears to be silent or very quiet")
            return False
        
        return True
    
    def test_microphone(self) -> bool:
        """
        Test microphone functionality.
        
        Returns:
            True if microphone is working
        """
        try:
            print("🧪 Testing microphone...")
            
            # List available audio devices
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"   {i}: {device['name']} ({'input' if device['max_input_channels'] > 0 else 'output'})")
            
            # Record a short test
            test_audio = self.record_audio(1.0)
            
            if self.is_audio_valid(test_audio):
                print("✅ Microphone test passed")
                return True
            else:
                print("❌ Microphone test failed - no valid audio detected")
                return False
                
        except Exception as e:
            print(f"❌ Microphone test error: {e}")
            return False