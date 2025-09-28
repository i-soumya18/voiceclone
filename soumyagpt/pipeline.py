"""
Pipeline orchestrator for SoumyaGPT.
Coordinates the STT -> Gemini -> TTS workflow for voice conversations.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .config import config
from .stt import SpeechToText
from .gemini import GeminiChat
from .tts import TextToSpeech


class VoiceChatPipeline:
    """Main pipeline that orchestrates voice conversation workflow."""
    
    def __init__(self):
        """Initialize the voice chat pipeline."""
        self.logger = logging.getLogger(__name__)
        self.stt = None
        self.gemini = None
        self.tts = None
        
        # Performance metrics
        self.metrics = {
            "total_conversations": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "successful_speeches": 0,
            "failed_speeches": 0,
            "average_response_time": 0.0,
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            print("🔧 Initializing SoumyaGPT components...")
            
            # Initialize Speech-to-Text
            print("   Loading STT (Whisper)...")
            self.stt = SpeechToText()
            
            # Initialize Gemini Chat
            print("   Loading Gemini API...")
            self.gemini = GeminiChat()
            
            # Initialize Text-to-Speech
            print("   Loading TTS (Coqui)...")
            self.tts = TextToSpeech()
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    def process_voice_input(
        self,
        duration: Optional[float] = None,
        language: str = "en",
        enable_voice_output: bool = True,
        save_audio: bool = False
    ) -> Optional[str]:
        """
        Process voice input through the complete pipeline.
        
        Args:
            duration: Recording duration in seconds
            language: Language for speech recognition
            enable_voice_output: Whether to enable voice response
            save_audio: Whether to save generated audio
            
        Returns:
            AI response text, or None if failed
        """
        start_time = time.time()
        
        try:
            # Step 1: Record and transcribe audio
            print("\n" + "="*60)
            print("🎤 STEP 1: Recording your voice...")
            
            transcription, confidence = self.stt.transcribe_microphone(duration, language)
            
            if not transcription:
                print("❌ No speech detected or transcription failed")
                self.metrics["failed_transcriptions"] += 1
                return None
            
            self.metrics["successful_transcriptions"] += 1
            print(f"👤 You said: {transcription}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Step 2: Get AI response
            print("\n🧠 STEP 2: Getting AI response...")
            
            response = self.gemini.get_response(transcription)
            
            if not response:
                print("❌ Failed to get AI response")
                self.metrics["failed_responses"] += 1
                return None
            
            self.metrics["successful_responses"] += 1
            print(f"🤖 SoumyaGPT: {response}")
            
            # Step 3: Convert to speech and play
            if enable_voice_output:
                print("\n🗣️  STEP 3: Converting to speech...")
                
                speech_success = self.tts.speak_text(
                    response, 
                    use_voice_clone=True, 
                    save_audio=save_audio
                )
                
                if speech_success:
                    self.metrics["successful_speeches"] += 1
                    print("🔊 Voice response played")
                else:
                    self.metrics["failed_speeches"] += 1
                    print("⚠️  Voice synthesis failed, but text response available")
            else:
                print("\n🔇 Voice output disabled")
            
            # Update metrics
            self.metrics["total_conversations"] += 1
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
            print(f"\n⏱️  Total processing time: {response_time:.2f}s")
            print("="*60)
            
            return response
            
        except KeyboardInterrupt:
            print("\n❌ Process interrupted by user")
            return None
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
    
    def process_audio_file(
        self,
        audio_file_path: str,
        language: str = "en",
        enable_voice_output: bool = True,
        save_audio: bool = False
    ) -> Optional[str]:
        """
        Process audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to audio file
            language: Language for speech recognition
            enable_voice_output: Whether to enable voice response
            save_audio: Whether to save generated audio
            
        Returns:
            AI response text, or None if failed
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and transcribe audio file
            print("\n" + "="*60)
            print(f"📁 STEP 1: Processing audio file: {audio_file_path}")
            
            transcription, confidence = self.stt.transcribe_file(audio_file_path, language)
            
            if not transcription:
                print("❌ Transcription failed")
                self.metrics["failed_transcriptions"] += 1
                return None
            
            self.metrics["successful_transcriptions"] += 1
            print(f"👤 Transcription: {transcription}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Step 2: Get AI response
            print("\n🧠 STEP 2: Getting AI response...")
            
            response = self.gemini.get_response(transcription)
            
            if not response:
                print("❌ Failed to get AI response")
                self.metrics["failed_responses"] += 1
                return None
            
            self.metrics["successful_responses"] += 1
            print(f"🤖 SoumyaGPT: {response}")
            
            # Step 3: Convert to speech and play
            if enable_voice_output:
                print("\n🗣️  STEP 3: Converting to speech...")
                
                speech_success = self.tts.speak_text(
                    response, 
                    use_voice_clone=True, 
                    save_audio=save_audio
                )
                
                if speech_success:
                    self.metrics["successful_speeches"] += 1
                    print("🔊 Voice response played")
                else:
                    self.metrics["failed_speeches"] += 1
                    print("⚠️  Voice synthesis failed, but text response available")
            else:
                print("\n🔇 Voice output disabled")
            
            # Update metrics
            self.metrics["total_conversations"] += 1
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
            print(f"\n⏱️  Total processing time: {response_time:.2f}s")
            print("="*60)
            
            return response
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
    
    def process_text_input(
        self,
        text: str,
        enable_voice_output: bool = True,
        save_audio: bool = False
    ) -> Optional[str]:
        """
        Process text input (skip STT step).
        
        Args:
            text: Input text
            enable_voice_output: Whether to enable voice response
            save_audio: Whether to save generated audio
            
        Returns:
            AI response text, or None if failed
        """
        start_time = time.time()
        
        try:
            print("\n" + "="*60)
            print(f"💬 Processing text input: {text}")
            
            # Step 1: Get AI response
            print("\n🧠 Getting AI response...")
            
            response = self.gemini.get_response(text)
            
            if not response:
                print("❌ Failed to get AI response")
                self.metrics["failed_responses"] += 1
                return None
            
            self.metrics["successful_responses"] += 1
            print(f"🤖 SoumyaGPT: {response}")
            
            # Step 2: Convert to speech and play
            if enable_voice_output:
                print("\n🗣️  Converting to speech...")
                
                speech_success = self.tts.speak_text(
                    response, 
                    use_voice_clone=True, 
                    save_audio=save_audio
                )
                
                if speech_success:
                    self.metrics["successful_speeches"] += 1
                    print("🔊 Voice response played")
                else:
                    self.metrics["failed_speeches"] += 1
                    print("⚠️  Voice synthesis failed, but text response available")
            else:
                print("\n🔇 Voice output disabled")
            
            # Update metrics
            response_time = time.time() - start_time
            
            print(f"\n⏱️  Total processing time: {response_time:.2f}s")
            print("="*60)
            
            return response
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
    
    def test_all_components(self) -> bool:
        """
        Test all pipeline components.
        
        Returns:
            True if all tests pass
        """
        print("🧪 Testing all SoumyaGPT components...")
        print("=" * 50)
        
        all_passed = True
        
        # Test STT
        try:
            print("\n1. Testing Speech-to-Text (Whisper)...")
            if self.stt and self.stt.test_microphone():
                print("   ✅ STT: PASSED")
            else:
                print("   ❌ STT: FAILED")
                all_passed = False
        except Exception as e:
            print(f"   ❌ STT: FAILED ({e})")
            all_passed = False
        
        # Test Gemini
        try:
            print("\n2. Testing Gemini API...")
            if self.gemini and self.gemini.test_connection():
                print("   ✅ Gemini: PASSED")
            else:
                print("   ❌ Gemini: FAILED")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Gemini: FAILED ({e})")
            all_passed = False
        
        # Test TTS
        try:
            print("\n3. Testing Text-to-Speech (Coqui)...")
            if self.tts and self.tts.test_tts():
                print("   ✅ TTS: PASSED")
            else:
                print("   ❌ TTS: FAILED")
                all_passed = False
        except Exception as e:
            print(f"   ❌ TTS: FAILED ({e})")
            all_passed = False
        
        # Test Voice Cloning
        try:
            print("\n4. Testing Voice Cloning...")
            if self.tts and config.reference_audio_path.exists():
                if self.tts.test_voice_clone():
                    print("   ✅ Voice Clone: PASSED")
                else:
                    print("   ⚠️  Voice Clone: PARTIAL (default voice works)")
                    # Don't fail overall test for voice cloning issues
            else:
                print("   ⚠️  Voice Clone: SKIPPED (no reference audio)")
        except Exception as e:
            print(f"   ⚠️  Voice Clone: FAILED ({e})")
            # Don't fail overall test for voice cloning issues
        
        print("\n" + "=" * 50)
        
        if all_passed:
            print("🎉 All critical components passed!")
            print("   Ready for voice conversations!")
        else:
            print("❌ Some components failed")
            print("   Check configuration and try again")
        
        return all_passed
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric."""
        current_avg = self.metrics["average_response_time"]
        total_convs = self.metrics["total_conversations"]
        
        if total_convs == 1:
            self.metrics["average_response_time"] = response_time
        else:
            # Calculate running average
            self.metrics["average_response_time"] = (
                (current_avg * (total_convs - 1)) + response_time
            ) / total_convs
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()
    
    def print_metrics(self):
        """Print performance metrics."""
        print("\n📊 SoumyaGPT Performance Metrics")
        print("=" * 40)
        
        m = self.metrics
        
        print(f"Total Conversations: {m['total_conversations']}")
        print(f"Average Response Time: {m['average_response_time']:.2f}s")
        print(f"")
        print(f"Transcription Success Rate: {self._get_success_rate('transcriptions')}%")
        print(f"AI Response Success Rate: {self._get_success_rate('responses')}%") 
        print(f"Speech Synthesis Success Rate: {self._get_success_rate('speeches')}%")
        print(f"")
        print(f"Component Status:")
        print(f"  STT (Whisper): {'✅' if self.stt else '❌'}")
        print(f"  Gemini API: {'✅' if self.gemini else '❌'}")
        print(f"  TTS (Coqui): {'✅' if self.tts else '❌'}")
        print(f"  Voice Clone: {'✅' if config.reference_audio_path.exists() else '❌'}")
    
    def _get_success_rate(self, component: str) -> float:
        """Calculate success rate for a component."""
        successful = self.metrics[f"successful_{component}"]
        failed = self.metrics[f"failed_{component}"]
        total = successful + failed
        
        if total == 0:
            return 0.0
        
        return (successful / total) * 100
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            "total_conversations": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "successful_speeches": 0,
            "failed_speeches": 0,
            "average_response_time": 0.0,
        }
        print("📊 Metrics reset")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        try:
            print("🔄 Shutting down SoumyaGPT...")
            
            # Print final metrics
            if self.metrics["total_conversations"] > 0:
                self.print_metrics()
            
            # Clean up components
            if hasattr(self.gemini, 'chat') and self.gemini.chat:
                # Save conversation history if needed
                pass
            
            print("✅ Shutdown complete")
            
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()