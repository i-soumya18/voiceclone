"""
Text-to-Speech module with voice synthesis for Python 3.13 compatibility.
Uses edge-tts and pyttsx3 as alternatives to Coqui TTS.
"""

import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import asyncio
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("⚠️  pyttsx3 not installed. Run: pip install pyttsx3")

try:
    import edge_tts
except ImportError:
    edge_tts = None
    print("⚠️  edge-tts not installed. Run: pip install edge-tts")

from .config import config


class TextToSpeech:
    """Text-to-Speech processor with multiple engine support for Python 3.13."""
    
    def __init__(self):
        self.pyttsx3_engine = None
        self.sample_rate = config.sample_rate
        self.logger = logging.getLogger(__name__)
        self.edge_voices = None
        
        # Initialize TTS engines
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize the TTS engines."""
        try:
            print("🔄 Loading TTS engines...")
            
            # Initialize pyttsx3 (offline TTS)
            if pyttsx3:
                try:
                    self.pyttsx3_engine = pyttsx3.init()
                    # Configure pyttsx3 settings
                    self.pyttsx3_engine.setProperty('rate', 150)  # Speed
                    self.pyttsx3_engine.setProperty('volume', 0.9)  # Volume
                    print("✅ pyttsx3 engine initialized")
                except Exception as e:
                    print(f"⚠️  pyttsx3 initialization failed: {e}")
                    self.pyttsx3_engine = None
            
            # Check Edge TTS availability
            if edge_tts:
                print("✅ Edge TTS available")
            
            if not self.pyttsx3_engine and not edge_tts:
                print("❌ No TTS engines available")
                print("💡 Install with: pip install pyttsx3 edge-tts")
            else:
                print("✅ TTS system initialized")
                
        except Exception as e:
            print(f"❌ Error loading TTS: {e}")
            self.pyttsx3_engine = None
    
    async def _synthesize_with_edge_tts(self, text: str, output_path: str, voice: str = "en-US-AriaNeural") -> bool:
        """Synthesize speech using Edge TTS."""
        try:
            if not edge_tts:
                return False
            
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return True
            
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
            return False
    
    def _synthesize_with_pyttsx3(self, text: str, output_path: str) -> bool:
        """Synthesize speech using pyttsx3."""
        try:
            if not self.pyttsx3_engine:
                return False
            
            # Save to file using pyttsx3
            self.pyttsx3_engine.save_to_file(text, output_path)
            self.pyttsx3_engine.runAndWait()
            
            # Check if file was created
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                # Fallback: use direct speech (no file output)
                print("⚠️  File save failed, using direct speech")
                return False
                
        except Exception as e:
            print(f"❌ pyttsx3 error: {e}")
            return False
    
    def synthesize_speech(
        self, 
        text: str, 
        output_path: Optional[Union[str, Path]] = None,
        use_voice_clone: bool = True,
        voice: str = "en-US-AriaNeural"
    ) -> Optional[str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file (optional)
            use_voice_clone: For compatibility (not used with current engines)
            voice: Voice to use for Edge TTS
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not text or not text.strip():
            print("⚠️  Empty text provided for TTS")
            return None
        
        try:
            print(f"🗣️  Synthesizing speech: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Prepare output path
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            output_path = str(Path(output_path))
            
            # Try Edge TTS first (better quality)
            if edge_tts:
                print("� Using Edge TTS (online)...")
                try:
                    # Run Edge TTS in async context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        self._synthesize_with_edge_tts(text, output_path, voice)
                    )
                    loop.close()
                    
                    if success and Path(output_path).exists():
                        print(f"✅ Speech synthesized with Edge TTS: {output_path}")
                        return output_path
                except Exception as e:
                    print(f"⚠️  Edge TTS failed: {e}")
            
            # Fallback to pyttsx3
            if self.pyttsx3_engine:
                print("💻 Using pyttsx3 (offline)...")
                if self._synthesize_with_pyttsx3(text, output_path):
                    print(f"✅ Speech synthesized with pyttsx3: {output_path}")
                    return output_path
            
            print("❌ All TTS engines failed")
            return None
                
        except Exception as e:
            print(f"❌ Error during speech synthesis: {e}")
            return None
    
    def play_audio(self, audio_path: Union[str, Path]) -> bool:
        """
        Play audio file through speakers.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if playback was successful
        """
        try:
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                print(f"❌ Audio file not found: {audio_path}")
                return False
            
            print(f"🔊 Playing audio: {audio_path}")
            
            # Load and play audio
            audio, sr = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Play audio
            sd.play(audio, sr)
            sd.wait()  # Wait until audio finishes playing
            
            print("✅ Audio playback completed")
            return True
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            return False
    
    def speak_text(
        self, 
        text: str, 
        use_voice_clone: bool = True, 
        save_audio: bool = False,
        output_dir: Optional[Path] = None,
        voice: str = "en-US-AriaNeural"
    ) -> bool:
        """
        Convert text to speech and play it immediately.
        
        Args:
            text: Text to speak
            use_voice_clone: For compatibility (not used with current engines)
            save_audio: Whether to save the audio file
            output_dir: Directory to save audio (if save_audio is True)
            voice: Voice to use for Edge TTS
            
        Returns:
            True if successful
        """
        try:
            # Generate speech
            temp_audio_path = self.synthesize_speech(text, voice=voice)
            
            if not temp_audio_path:
                # Fallback: try direct pyttsx3 speech
                if self.pyttsx3_engine:
                    print("🔄 Trying direct speech output...")
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                    return True
                return False
            
            # Play the audio
            success = self.play_audio(temp_audio_path)
            
            # Save audio if requested
            if save_audio and success:
                if output_dir is None:
                    output_dir = config.data_dir / "generated_speech"
                    output_dir.mkdir(exist_ok=True)
                
                import time
                timestamp = int(time.time())
                saved_path = output_dir / f"speech_{timestamp}.wav"
                
                Path(temp_audio_path).rename(saved_path)
                print(f"💾 Audio saved: {saved_path}")
            else:
                # Clean up temporary file
                Path(temp_audio_path).unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            print(f"❌ Error in speak_text: {e}")
            return False
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices.
        
        Returns:
            Dictionary of available voices
        """
        voices = {}
        
        # pyttsx3 voices
        if self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                for i, voice in enumerate(pyttsx3_voices):
                    voices[f"pyttsx3_{i}"] = {
                        "name": voice.name,
                        "id": voice.id,
                        "language": getattr(voice, 'languages', ['en']),
                        "engine": "pyttsx3"
                    }
            except Exception as e:
                print(f"⚠️  Could not get pyttsx3 voices: {e}")
        
        # Edge TTS voices (common ones)
        if edge_tts:
            edge_voices = {
                "aria": {"name": "Aria (US Female)", "id": "en-US-AriaNeural", "engine": "edge-tts"},
                "guy": {"name": "Guy (US Male)", "id": "en-US-GuyNeural", "engine": "edge-tts"},
                "jenny": {"name": "Jenny (US Female)", "id": "en-US-JennyNeural", "engine": "edge-tts"},
                "davis": {"name": "Davis (US Male)", "id": "en-US-DavisNeural", "engine": "edge-tts"},
            }
            voices.update(edge_voices)
        
        return voices
    
    def test_tts(self, test_text: str = "Hello, this is a test of the text to speech system.") -> bool:
        """
        Test TTS functionality.
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            True if TTS is working
        """
        try:
            print("🧪 Testing TTS functionality...")
            
            if not self.pyttsx3_engine and not edge_tts:
                print("❌ No TTS engines available")
                return False
            
            # Test synthesis
            success = self.speak_text(test_text, use_voice_clone=False)
            
            if success:
                print("✅ TTS test passed")
                return True
            else:
                print("❌ TTS test failed")
                return False
                
        except Exception as e:
            print(f"❌ TTS test error: {e}")
            return False
    
    def test_voice_clone(self, test_text: str = "This is a test of voice synthesis.") -> bool:
        """
        Test voice synthesis functionality.
        Note: True voice cloning not available with current engines.
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            True if voice synthesis is working
        """
        try:
            print("🧪 Testing voice synthesis functionality...")
            print("ℹ️  Note: True voice cloning requires Coqui TTS (Python <3.12)")
            
            if not self.pyttsx3_engine and not edge_tts:
                print("❌ No TTS engines available")
                return False
            
            # Test with different voice if available
            voices = self.get_available_voices()
            if voices:
                # Try first available voice
                first_voice = list(voices.keys())[0]
                voice_info = voices[first_voice]
                print(f"🎤 Testing with voice: {voice_info.get('name', first_voice)}")
                
                if voice_info.get('engine') == 'edge-tts':
                    success = self.speak_text(test_text, voice=voice_info['id'])
                else:
                    # pyttsx3 voice
                    if self.pyttsx3_engine:
                        self.pyttsx3_engine.setProperty('voice', voice_info['id'])
                    success = self.speak_text(test_text)
            else:
                success = self.speak_text(test_text)
            
            if success:
                print("✅ Voice synthesis test passed")
                return True
            else:
                print("❌ Voice synthesis test failed")
                return False
                
        except Exception as e:
            print(f"❌ Voice synthesis test error: {e}")
            return False
    
    def setup_voice_samples(self) -> bool:
        """
        Guide user through setting up voice samples for future voice cloning.
        
        Returns:
            True if setup guidance is complete
        """
        print("🎭 Voice Synthesis Setup")
        print("=" * 50)
        
        print("ℹ️  Current TTS engines support multiple voices but not custom voice cloning.")
        print("   For true voice cloning, Coqui TTS is required (Python <3.12)")
        print("")
        print("📋 Available options:")
        print("1. Use high-quality Edge TTS voices (recommended)")
        print("2. Use offline pyttsx3 voices")
        print("3. For voice cloning: Use Python 3.11 environment with Coqui TTS")
        print("")
        
        # Show available voices
        voices = self.get_available_voices()
        if voices:
            print("🎤 Available voices:")
            for voice_id, voice_info in voices.items():
                engine = voice_info.get('engine', 'unknown')
                name = voice_info.get('name', voice_id)
                print(f"   - {name} ({engine})")
        
        return True
    
    def record_reference_voice(self, duration: float = 20.0) -> bool:
        """
        Record reference voice for future voice cloning setup.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            True if recording was successful
        """
        try:
            print(f"🎤 Recording reference voice for {duration} seconds...")
            print("   This will be saved for future voice cloning setup")
            print("   3... 2... 1... START!")
            
            # Record audio
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Save reference audio
            reference_path = config.reference_audio_path
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(reference_path), audio.flatten(), self.sample_rate)
            
            print(f"✅ Reference voice recorded: {reference_path}")
            print("💡 This can be used with Coqui TTS in a Python 3.11 environment")
            
            # Test current synthesis
            print("🧪 Testing current voice synthesis...")
            if self.test_tts("This is a test of the recorded voice setup."):
                print("🎉 Voice recording and synthesis setup complete!")
                return True
            else:
                print("⚠️  Voice synthesis test failed, but recording was saved.")
                return True  # Recording still successful
                
        except Exception as e:
            print(f"❌ Error recording reference voice: {e}")
            return False