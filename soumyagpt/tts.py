"""
Text-to-Speech module with voice synthesis and XTTS voice cloning support.
Uses edge-tts, pyttsx3, and Coqui XTTS for comprehensive voice synthesis.
"""

import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import asyncio
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
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

try:
    from .xtts_cloner import XTTSVoiceCloner
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    XTTSVoiceCloner = None
    print("⚠️  XTTS not available. Voice cloning features disabled.")

from .config import config


class TextToSpeech:
    """Text-to-Speech processor with multiple engine support and XTTS voice cloning."""
    
    def __init__(self):
        self.pyttsx3_engine = None
        self.sample_rate = config.sample_rate
        self.logger = logging.getLogger(__name__)
        self.edge_voices = None
        
        # Initialize XTTS voice cloner if available
        self.xtts_cloner = None
        if XTTS_AVAILABLE:
            try:
                self.xtts_cloner = XTTSVoiceCloner()
                print("✅ XTTS voice cloner initialized")
            except Exception as e:
                print(f"⚠️  XTTS initialization failed: {e}")
                self.xtts_cloner = None
        
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
            
            # Check XTTS availability
            if self.xtts_cloner:
                print("✅ XTTS voice cloning available")
            
            if not self.pyttsx3_engine and not edge_tts and not self.xtts_cloner:
                print("❌ No TTS engines available")
                print("💡 Install with: pip install pyttsx3 edge-tts TTS")
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
        voice: str = "en-US-AriaNeural",
        speaker_name: Optional[str] = None,
        reference_audio: Optional[str] = None
    ) -> Optional[str]:
        """
        Synthesize speech from text with multiple engine options.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file (optional)
            use_voice_clone: Whether to use XTTS voice cloning if available
            voice: Voice to use for Edge TTS
            speaker_name: Name of custom XTTS speaker model
            reference_audio: Reference audio for XTTS voice cloning
            
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
            
            # Try XTTS voice cloning first if requested and available
            if use_voice_clone and self.xtts_cloner:
                try:
                    # Try custom speaker model first
                    if speaker_name:
                        print(f"🎭 Using XTTS custom speaker: {speaker_name}")
                        result = self.xtts_cloner.synthesize_with_custom_voice(
                            text=text,
                            speaker_name=speaker_name,
                            language="en",
                            output_path=output_path
                        )
                        if result:
                            print(f"✅ Speech synthesized with XTTS custom voice: {output_path}")
                            return output_path
                    
                    # Try reference audio cloning
                    elif reference_audio and Path(reference_audio).exists():
                        print(f"🎭 Using XTTS with reference audio: {Path(reference_audio).name}")
                        result = self.xtts_cloner.clone_voice(
                            text=text,
                            speaker_wav=reference_audio,
                            language="en",
                            output_path=output_path
                        )
                        if result:
                            print(f"✅ Speech synthesized with XTTS voice cloning: {output_path}")
                            return output_path
                    
                    # Try default reference audio
                    elif config.reference_audio_path.exists():
                        print(f"🎭 Using XTTS with default reference audio")
                        result = self.xtts_cloner.clone_voice(
                            text=text,
                            speaker_wav=str(config.reference_audio_path),
                            language="en",
                            output_path=output_path
                        )
                        if result:
                            print(f"✅ Speech synthesized with XTTS voice cloning: {output_path}")
                            return output_path
                    
                    print("⚠️  XTTS voice cloning failed, falling back to other engines")
                    
                except Exception as e:
                    print(f"⚠️  XTTS error: {e}, falling back to other engines")
            
            # Try Edge TTS (better quality)
            if edge_tts:
                print("💫 Using Edge TTS (online)...")
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
        voice: str = "en-US-AriaNeural",
        speaker_name: Optional[str] = None,
        reference_audio: Optional[str] = None
    ) -> bool:
        """
        Convert text to speech and play it immediately.
        
        Args:
            text: Text to speak
            use_voice_clone: Whether to use voice cloning
            save_audio: Whether to save the audio file
            output_dir: Directory to save audio (if save_audio is True)
            voice: Voice to use for Edge TTS
            speaker_name: Custom XTTS speaker model name
            reference_audio: Reference audio for XTTS
            
        Returns:
            True if successful
        """
        try:
            # Generate speech
            temp_audio_path = self.synthesize_speech(
                text=text, 
                use_voice_clone=use_voice_clone,
                voice=voice,
                speaker_name=speaker_name,
                reference_audio=reference_audio
            )
            
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
    
    # XTTS Voice Cloning Methods
    
    def load_xtts_model(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> bool:
        """Load XTTS model for voice cloning."""
        if not self.xtts_cloner:
            print("❌ XTTS not available")
            return False
        
        return self.xtts_cloner.load_pretrained_model(model_name)
    
    def clone_voice_from_audio(self, text: str, reference_audio: str, output_path: Optional[str] = None) -> Optional[str]:
        """Clone voice using reference audio."""
        if not self.xtts_cloner:
            print("❌ XTTS not available")
            return None
        
        return self.xtts_cloner.clone_voice(text, reference_audio, output_path=output_path)
    
    def create_speaker_model(self, speaker_name: str, audio_files: List[str], transcripts: Optional[List[str]] = None) -> Optional[str]:
        """Create a custom speaker model from audio samples."""
        if not self.xtts_cloner:
            print("❌ XTTS not available")
            return None
        
        # Prepare training data
        dataset_path = self.xtts_cloner.prepare_training_data(speaker_name, audio_files, transcripts)
        if not dataset_path:
            return None
        
        # Fine-tune model (create speaker embeddings)
        return self.xtts_cloner.fine_tune_model(dataset_path, speaker_name)
    
    def load_speaker_model(self, model_path: str) -> bool:
        """Load a custom speaker model."""
        if not self.xtts_cloner:
            print("❌ XTTS not available")
            return False
        
        return self.xtts_cloner.load_speaker_model(model_path)
    
    def list_speaker_models(self) -> List[Dict]:
        """List available custom speaker models."""
        if not self.xtts_cloner:
            return []
        
        return self.xtts_cloner.list_speaker_models()
    
    def synthesize_with_speaker(self, text: str, speaker_name: str, output_path: Optional[str] = None) -> Optional[str]:
        """Synthesize speech with a custom speaker model."""
        if not self.xtts_cloner:
            print("❌ XTTS not available")
            return None
        
        return self.xtts_cloner.synthesize_with_custom_voice(text, speaker_name, output_path=output_path)
    
    # Existing methods continue...
    
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
        
        # XTTS custom voices
        if self.xtts_cloner:
            speaker_models = self.list_speaker_models()
            for model in speaker_models:
                speaker_name = model.get("speaker_name", "unknown")
                voices[f"xtts_{speaker_name}"] = {
                    "name": f"XTTS {speaker_name}",
                    "id": speaker_name,
                    "engine": "xtts",
                    "type": "custom"
                }
        
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
            
            if not self.pyttsx3_engine and not edge_tts and not self.xtts_cloner:
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
    
    def test_voice_clone(self, test_text: str = "This is a test of voice cloning technology.") -> bool:
        """
        Test voice cloning functionality.
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            True if voice cloning is working
        """
        try:
            print("🧪 Testing voice cloning functionality...")
            
            if not self.xtts_cloner:
                print("⚠️  XTTS not available, testing alternative voice synthesis")
                return self.test_tts(test_text)
            
            # Load XTTS model if not loaded
            if not self.xtts_cloner.model:
                print("🔄 Loading XTTS model...")
                if not self.load_xtts_model():
                    print("❌ Failed to load XTTS model")
                    return self.test_tts(test_text)
            
            # Test with default reference audio if available
            if config.reference_audio_path.exists():
                print(f"🎭 Testing voice cloning with reference audio")
                result = self.clone_voice_from_audio(test_text, str(config.reference_audio_path))
                if result:
                    success = self.play_audio(result)
                    # Clean up test file
                    Path(result).unlink(missing_ok=True)
                    if success:
                        print("✅ Voice cloning test passed")
                        return True
            
            # Test with custom speakers if available
            speaker_models = self.list_speaker_models()
            if speaker_models:
                speaker_name = speaker_models[0].get("speaker_name")
                print(f"🎭 Testing custom speaker: {speaker_name}")
                result = self.synthesize_with_speaker(test_text, speaker_name)
                if result:
                    success = self.play_audio(result)
                    # Clean up test file
                    Path(result).unlink(missing_ok=True)
                    if success:
                        print("✅ Custom speaker test passed")
                        return True
            
            print("⚠️  No reference audio or custom speakers available")
            print("💡 Record reference audio or train a speaker model to test voice cloning")
            return self.test_tts(test_text)
                
        except Exception as e:
            print(f"❌ Voice cloning test error: {e}")
            return False
    
    def setup_voice_cloning(self) -> bool:
        """
        Guide user through setting up voice cloning.
        
        Returns:
            True if setup guidance is complete
        """
        print("🎭 Voice Cloning Setup")
        print("=" * 50)
        
        if not self.xtts_cloner:
            print("❌ XTTS not available. Voice cloning requires Python 3.10/3.11.")
            print("💡 Current features available:")
            print("   - High-quality Edge TTS voices")
            print("   - Offline pyttsx3 voices")
            return False
        
        print("✅ XTTS voice cloning is available!")
        print("")
        print("📋 Voice cloning options:")
        print("1. Quick clone: Use a single reference audio file")
        print("2. Custom speaker: Train a model with multiple samples")
        print("3. Load existing speaker model")
        print("")
        
        # Check for existing reference audio
        if config.reference_audio_path.exists():
            print(f"✅ Reference audio found: {config.reference_audio_path}")
        else:
            print("⚠️  No reference audio found")
            print(f"💡 Record reference audio and save as: {config.reference_audio_path}")
        
        # List existing speaker models
        speaker_models = self.list_speaker_models()
        if speaker_models:
            print(f"\n🎤 Available speaker models ({len(speaker_models)}):")
            for model in speaker_models:
                name = model.get("speaker_name", "unknown")
                status = model.get("status", "saved")
                print(f"   - {name} ({status})")
        else:
            print("\n⚠️  No custom speaker models found")
            print("💡 Create speaker models using the CLI: soumyagpt record-voice")
        
        return True
    
    def record_reference_voice(self, duration: float = 20.0) -> bool:
        """
        Record reference voice for voice cloning.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            True if recording was successful
        """
        try:
            print(f"🎤 Recording reference voice for {duration} seconds...")
            print("   This will be used for XTTS voice cloning")
            print("   Speak clearly and naturally")
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
            
            # Test voice cloning
            if self.xtts_cloner:
                print("🧪 Testing voice cloning with new reference...")
                test_result = self.test_voice_clone("Hello, this is my cloned voice speaking.")
                if test_result:
                    print("🎉 Voice cloning setup complete!")
                    return True
                else:
                    print("⚠️  Voice cloning test failed, but recording was saved.")
                    return True
            else:
                print("💡 XTTS not available, but reference audio saved for future use")
                return True
                
        except Exception as e:
            print(f"❌ Error recording reference voice: {e}")
            return False