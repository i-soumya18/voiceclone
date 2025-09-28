"""
Command-line interface for SoumyaGPT voice-cloned chatbot.
Provides easy-to-use commands for voice conversations.
"""

import click
import sys
from pathlib import Path
from typing import Optional

# Import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
    
    class Style:
        RESET_ALL = ""
        BRIGHT = ""

from .config import config
from .pipeline import VoiceChatPipeline
from .voice_recorder import VoiceRecorder


def print_banner():
    """Print SoumyaGPT banner."""
    banner = f"""
{Fore.CYAN}🎙️  SoumyaGPT – Voice-Cloned CLI Chatbot{Style.RESET_ALL}
{Fore.YELLOW}═══════════════════════════════════════════════════════════════{Style.RESET_ALL}
{Fore.WHITE}Your AI twin that speaks in your voice using Gemini API{Style.RESET_ALL}
{Fore.WHITE}Hardware: GTX 1650 + i5 + 24GB RAM optimized{Style.RESET_ALL}
"""
    print(banner)


def print_status():
    """Print system status."""
    print(f"\n{Fore.CYAN}System Status:{Style.RESET_ALL}")
    
    # Check configuration
    gemini_status = "✅ Ready" if config.gemini_api_key else "❌ Not configured"
    print(f"  Gemini API: {gemini_status}")
    
    # Check reference audio
    reference_status = "✅ Available" if config.reference_audio_path.exists() else "❌ Missing"
    print(f"  Voice Clone: {reference_status}")
    
    # Show paths
    print(f"  Data Dir: {config.data_dir}")
    print(f"  Voice Samples: {config.voice_samples_dir}")


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--show-config", is_flag=True, help="Show configuration details")
@click.option("--status", is_flag=True, help="Show system status")
@click.pass_context
def main(ctx, version, show_config, status):
    """SoumyaGPT - Voice-cloned CLI chatbot using Gemini API."""
    
    if version:
        from . import __version__
        click.echo(f"SoumyaGPT version {__version__}")
        return
    
    if show_config:
        print_banner()
        config.print_config()
        return
    
    if status:
        print_banner()
        print_status()
        return
    
    # If no command provided, show help or start interactive mode
    if ctx.invoked_subcommand is None:
        print_banner()
        print_status()
        click.echo(f"\n{Fore.YELLOW}Use 'soumyagpt --help' for available commands{Style.RESET_ALL}")
        click.echo(f"{Fore.GREEN}Quick start: soumyagpt chat --mic{Style.RESET_ALL}")


@main.command()
@click.option("--mic", is_flag=True, help="Use microphone input")
@click.option("--file", "audio_file", type=click.Path(exists=True), help="Use audio file input")
@click.option("--duration", type=float, default=None, help="Recording duration in seconds")
@click.option("--continuous", is_flag=True, help="Continuous conversation mode")
@click.option("--no-voice", is_flag=True, help="Disable voice output (text only)")
@click.option("--save-audio", is_flag=True, help="Save generated speech to files")
@click.option("--language", default="en", help="Language for speech recognition")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def chat(mic, audio_file, duration, continuous, no_voice, save_audio, language, verbose):
    """Start a voice conversation with SoumyaGPT."""
    
    print_banner()
    
    # Set verbose mode
    if verbose:
        config.verbose = True
    
    # Validate inputs
    if not mic and not audio_file:
        click.echo(f"{Fore.RED}❌ Please specify either --mic or --file{Style.RESET_ALL}")
        click.echo(f"{Fore.YELLOW}Example: soumyagpt chat --mic{Style.RESET_ALL}")
        return
    
    if mic and audio_file:
        click.echo(f"{Fore.RED}❌ Please use either --mic OR --file, not both{Style.RESET_ALL}")
        return
    
    # Initialize pipeline
    try:
        pipeline = VoiceChatPipeline()
        
        # Test components
        print(f"\n{Fore.CYAN}🔧 Initializing components...{Style.RESET_ALL}")
        
        if not pipeline.test_all_components():
            click.echo(f"{Fore.RED}❌ Component initialization failed{Style.RESET_ALL}")
            return
        
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error initializing SoumyaGPT: {e}{Style.RESET_ALL}")
        return
    
    # Start conversation
    try:
        if mic:
            if continuous:
                _continuous_mic_chat(pipeline, duration, not no_voice, save_audio, language)
            else:
                _single_mic_chat(pipeline, duration, not no_voice, save_audio, language)
        else:
            _file_chat(pipeline, audio_file, not no_voice, save_audio, language)
            
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}👋 Conversation ended by user{Style.RESET_ALL}")
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error during conversation: {e}{Style.RESET_ALL}")


@main.command()
@click.option("--duration", type=float, default=20.0, help="Recording duration in seconds")
def setup_voice(duration):
    """Set up voice cloning by recording your voice."""
    
    print_banner()
    print(f"\n{Fore.CYAN}🎭 Voice Cloning Setup{Style.RESET_ALL}")
    
    try:
        from .tts import TextToSpeech
        
        tts = TextToSpeech()
        
        if tts.record_reference_voice(duration):
            click.echo(f"{Fore.GREEN}🎉 Voice cloning setup complete!{Style.RESET_ALL}")
        else:
            click.echo(f"{Fore.RED}❌ Voice setup failed{Style.RESET_ALL}")
            
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error setting up voice: {e}{Style.RESET_ALL}")


@main.command()
@click.argument("text", required=True)
@click.option("--no-clone", is_flag=True, help="Use default voice instead of cloning")
@click.option("--save", is_flag=True, help="Save audio file")
def speak(text, no_clone, save):
    """Convert text to speech using your cloned voice."""
    
    try:
        from .tts import TextToSpeech
        
        tts = TextToSpeech()
        use_clone = not no_clone
        
        print(f"🗣️  Speaking: {text}")
        
        success = tts.speak_text(text, use_voice_clone=use_clone, save_audio=save)
        
        if success:
            click.echo(f"{Fore.GREEN}✅ Speech completed{Style.RESET_ALL}")
        else:
            click.echo(f"{Fore.RED}❌ Speech failed{Style.RESET_ALL}")
            
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


@main.command()
@click.option("--stt", is_flag=True, help="Test speech-to-text")
@click.option("--tts", is_flag=True, help="Test text-to-speech")
@click.option("--gemini", is_flag=True, help="Test Gemini API")
@click.option("--voice-clone", is_flag=True, help="Test voice cloning")
@click.option("--all", "test_all", is_flag=True, help="Test all components")
def test(stt, tts, gemini, voice_clone, test_all):
    """Test individual components or the entire system."""
    
    print_banner()
    
    try:
        from .pipeline import VoiceChatPipeline
        
        pipeline = VoiceChatPipeline()
        
        if test_all:
            print(f"\n{Fore.CYAN}🧪 Testing all components...{Style.RESET_ALL}")
            success = pipeline.test_all_components()
            
            if success:
                click.echo(f"{Fore.GREEN}✅ All tests passed!{Style.RESET_ALL}")
            else:
                click.echo(f"{Fore.RED}❌ Some tests failed{Style.RESET_ALL}")
            return
        
        # Test individual components
        if stt:
            print(f"\n{Fore.CYAN}🎤 Testing Speech-to-Text...{Style.RESET_ALL}")
            success = pipeline.stt.test_microphone()
            status = "✅ PASSED" if success else "❌ FAILED"
            click.echo(f"STT Test: {status}")
        
        if tts:
            print(f"\n{Fore.CYAN}🗣️  Testing Text-to-Speech...{Style.RESET_ALL}")
            success = pipeline.tts.test_tts()
            status = "✅ PASSED" if success else "❌ FAILED"
            click.echo(f"TTS Test: {status}")
        
        if gemini:
            print(f"\n{Fore.CYAN}🧠 Testing Gemini API...{Style.RESET_ALL}")
            success = pipeline.gemini.test_connection()
            status = "✅ PASSED" if success else "❌ FAILED"
            click.echo(f"Gemini Test: {status}")
        
        if voice_clone:
            print(f"\n{Fore.CYAN}🎭 Testing Voice Cloning...{Style.RESET_ALL}")
            success = pipeline.tts.test_voice_clone()
            status = "✅ PASSED" if success else "❌ FAILED"
            click.echo(f"Voice Clone Test: {status}")
        
        # If no specific test requested, show help
        if not any([stt, tts, gemini, voice_clone]):
            click.echo(f"{Fore.YELLOW}Use --all to test everything, or specify individual components{Style.RESET_ALL}")
            click.echo(f"Available: --stt, --tts, --gemini, --voice-clone")
            
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error during testing: {e}{Style.RESET_ALL}")


@main.command()
@click.option("--reset", is_flag=True, help="Reset conversation history")
@click.option("--save", type=click.Path(), help="Save conversation to file")
@click.option("--load", type=click.Path(exists=True), help="Load conversation from file")
def history(reset, save, load):
    """Manage conversation history."""
    
    try:
        from .gemini import GeminiChat
        
        gemini = GeminiChat()
        
        if reset:
            gemini.reset_conversation()
            click.echo(f"{Fore.GREEN}✅ Conversation history reset{Style.RESET_ALL}")
        
        if save:
            gemini.save_conversation(save)
            click.echo(f"{Fore.GREEN}✅ Conversation saved to {save}{Style.RESET_ALL}")
        
        if load:
            gemini.load_conversation(load)
            click.echo(f"{Fore.GREEN}✅ Conversation loaded from {load}{Style.RESET_ALL}")
        
        # Show current history length
        history_len = len(gemini.get_conversation_history())
        click.echo(f"Current conversation length: {history_len} messages")
        
    except Exception as e:
        click.echo(f"{Fore.RED}❌ Error managing history: {e}{Style.RESET_ALL}")


def _single_mic_chat(pipeline, duration, enable_voice, save_audio, language):
    """Handle single microphone conversation."""
    
    print(f"\n{Fore.GREEN}🎤 Single conversation mode{Style.RESET_ALL}")
    print(f"Recording duration: {duration or config.default_duration} seconds")
    print(f"Language: {language}")
    print(f"Voice output: {'enabled' if enable_voice else 'disabled'}")
    print(f"\nPress Ctrl+C to exit\n")
    
    # Single conversation
    response = pipeline.process_voice_input(
        duration=duration,
        language=language,
        enable_voice_output=enable_voice,
        save_audio=save_audio
    )
    
    if response:
        print(f"\n{Fore.CYAN}Conversation completed successfully{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Conversation failed{Style.RESET_ALL}")


def _continuous_mic_chat(pipeline, duration, enable_voice, save_audio, language):
    """Handle continuous microphone conversation."""
    
    print(f"\n{Fore.GREEN}🔄 Continuous conversation mode{Style.RESET_ALL}")
    print(f"Recording duration: {duration or config.default_duration} seconds per turn")
    print(f"Language: {language}")
    print(f"Voice output: {'enabled' if enable_voice else 'disabled'}")
    print(f"\nPress Ctrl+C to exit, or say 'goodbye' to end\n")
    
    conversation_count = 0
    
    while True:
        try:
            print(f"{Fore.MAGENTA}--- Turn {conversation_count + 1} ---{Style.RESET_ALL}")
            
            response = pipeline.process_voice_input(
                duration=duration,
                language=language,
                enable_voice_output=enable_voice,
                save_audio=save_audio
            )
            
            if not response:
                print(f"{Fore.YELLOW}⚠️  No response received. Try again.{Style.RESET_ALL}")
                continue
            
            conversation_count += 1
            
            # Check for exit conditions
            if any(word in response.lower() for word in ["goodbye", "bye", "exit", "quit", "stop"]):
                print(f"\n{Fore.YELLOW}👋 Ending conversation as requested{Style.RESET_ALL}")
                break
            
            print(f"\n{Fore.CYAN}Ready for next turn... (Ctrl+C to exit){Style.RESET_ALL}\n")
            
        except KeyboardInterrupt:
            break
    
    print(f"\n{Fore.GREEN}Conversation ended after {conversation_count} turns{Style.RESET_ALL}")


def _file_chat(pipeline, audio_file, enable_voice, save_audio, language):
    """Handle audio file conversation."""
    
    print(f"\n{Fore.GREEN}📁 Processing audio file: {audio_file}{Style.RESET_ALL}")
    print(f"Language: {language}")
    print(f"Voice output: {'enabled' if enable_voice else 'disabled'}")
    print("")
    
    response = pipeline.process_audio_file(
        audio_file,
        language=language,
        enable_voice_output=enable_voice,
        save_audio=save_audio
    )
    
    if response:
        print(f"\n{Fore.CYAN}File processing completed successfully{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}File processing failed{Style.RESET_ALL}")


if __name__ == "__main__":
    main()


# Voice Cloning Commands

@main.command("record-voice")
@click.option("--speaker", default="default", help="Speaker name for the voice model")
@click.option("--samples", type=int, default=10, help="Number of voice samples to record")
@click.option("--difficulty", type=click.Choice(["basic", "medium", "advanced"]), default="medium", help="Training sentence difficulty")
@click.option("--device", type=int, default=None, help="Audio device ID")
def record_voice(speaker, samples, difficulty, device):
    """Record voice samples for training a custom voice model."""
    
    print(f"{Fore.CYAN}🎙️  Voice Recording Session{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Speaker: {speaker}")
    print(f"Samples to record: {samples}")
    print(f"Difficulty: {difficulty}")
    print("")
    
    try:
        recorder = VoiceRecorder()
        
        # Test microphone first
        if not recorder.test_microphone(device_id=device):
            print(f"{Fore.RED}❌ Microphone test failed. Check your audio setup.{Style.RESET_ALL}")
            return
        
        # Get training sentences
        sentences = recorder.get_training_sentences(difficulty)
        selected_sentences = sentences[:samples]
        
        # Record training set
        recorded_files = recorder.record_training_set(
            sentences=selected_sentences,
            speaker_name=speaker,
            device_id=device
        )
        
        if recorded_files:
            print(f"\n{Fore.GREEN}✅ Voice recording complete!{Style.RESET_ALL}")
            print(f"Recorded {len(recorded_files)} samples")
            print(f"Use 'soumyagpt train-voice --speaker {speaker}' to train the model")
        else:
            print(f"{Fore.RED}❌ Voice recording failed{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Voice recording cancelled{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error during voice recording: {e}{Style.RESET_ALL}")


@main.command("train-voice")
@click.option("--speaker", required=True, help="Speaker name to train")
@click.option("--epochs", type=int, default=10, help="Training epochs")
@click.option("--batch-size", type=int, default=2, help="Training batch size")
def train_voice(speaker, epochs, batch_size):
    """Train a custom voice model from recorded samples."""
    
    print(f"{Fore.CYAN}🎯 Training Voice Model{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Speaker: {speaker}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("")
    
    try:
        from .tts import TextToSpeech
        tts = TextToSpeech()
        
        if not tts.xtts_cloner:
            print(f"{Fore.RED}❌ XTTS not available. Voice training requires Python 3.10/3.11.{Style.RESET_ALL}")
            return
        
        # Find recorded samples for the speaker
        voice_samples_dir = config.voice_samples_dir
        audio_files = list(voice_samples_dir.glob(f"{speaker}_sample_*.wav"))
        
        if not audio_files:
            print(f"{Fore.RED}❌ No voice samples found for speaker '{speaker}'{Style.RESET_ALL}")
            print(f"Record samples first: soumyagpt record-voice --speaker {speaker}")
            return
        
        print(f"Found {len(audio_files)} voice samples")
        
        # Create speaker model
        model_path = tts.create_speaker_model(
            speaker_name=speaker,
            audio_files=[str(f) for f in audio_files]
        )
        
        if model_path:
            print(f"\n{Fore.GREEN}✅ Voice model trained successfully!{Style.RESET_ALL}")
            print(f"Model saved: {model_path}")
            print(f"Test with: soumyagpt test-voice --speaker {speaker}")
        else:
            print(f"{Fore.RED}❌ Voice model training failed{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error during voice training: {e}{Style.RESET_ALL}")


@main.command("list-voices")
def list_voices():
    """List all available voices and models."""
    
    print(f"{Fore.CYAN}🎤 Available Voices{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        from .tts import TextToSpeech
        tts = TextToSpeech()
        
        voices = tts.get_available_voices()
        
        if not voices:
            print(f"{Fore.YELLOW}No voices available{Style.RESET_ALL}")
            return
        
        # Group voices by engine
        engines = {}
        for voice_id, voice_info in voices.items():
            engine = voice_info.get("engine", "unknown")
            if engine not in engines:
                engines[engine] = []
            engines[engine].append((voice_id, voice_info))
        
        for engine, voice_list in engines.items():
            print(f"\n{Fore.GREEN}{engine.upper()} Voices:{Style.RESET_ALL}")
            for voice_id, voice_info in voice_list:
                name = voice_info.get("name", voice_id)
                voice_type = voice_info.get("type", "")
                type_indicator = f" [{voice_type}]" if voice_type else ""
                print(f"  - {name}{type_indicator}")
        
        # List recorded samples
        recorder = VoiceRecorder()
        samples = recorder.list_recorded_samples()
        
        if samples:
            print(f"\n{Fore.BLUE}Recorded Voice Samples:{Style.RESET_ALL}")
            for sample in samples[:10]:  # Show first 10
                duration = sample['duration']
                size_kb = sample['size_kb']
                print(f"  - {sample['filename']} ({duration:.1f}s, {size_kb:.1f}KB)")
                
    except Exception as e:
        print(f"{Fore.RED}❌ Error listing voices: {e}{Style.RESET_ALL}")


@main.command("test-voice")
@click.option("--speaker", help="Test custom XTTS speaker")
@click.option("--voice", help="Test specific voice ID")
@click.option("--text", default="Hello, this is a test of my cloned voice speaking clearly.", help="Test text")
def test_voice(speaker, voice, text):
    """Test voice synthesis with different voices."""
    
    print(f"{Fore.CYAN}🧪 Testing Voice Synthesis{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Text: \"{text}\"")
    
    if speaker:
        print(f"Testing custom speaker: {speaker}")
    elif voice:
        print(f"Testing voice: {voice}")
    else:
        print("Testing default voice synthesis")
    
    print("")
    
    try:
        from .tts import TextToSpeech
        tts = TextToSpeech()
        
        # Test with custom speaker
        if speaker:
            if tts.xtts_cloner:
                result = tts.synthesize_with_speaker(text, speaker)
                if result:
                    success = tts.play_audio(result)
                    # Clean up
                    Path(result).unlink(missing_ok=True)
                    if success:
                        print(f"{Fore.GREEN}✅ Custom speaker test successful{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ Audio playback failed{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Custom speaker synthesis failed{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ XTTS not available for custom speakers{Style.RESET_ALL}")
        
        # Test with specific voice
        elif voice:
            success = tts.speak_text(text, voice=voice, use_voice_clone=False)
            if success:
                print(f"{Fore.GREEN}✅ Voice test successful{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Voice test failed{Style.RESET_ALL}")
        
        # Test default
        else:
            if tts.test_voice_clone(text):
                print(f"{Fore.GREEN}✅ Voice cloning test successful{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  Voice cloning not available, testing regular TTS{Style.RESET_ALL}")
                if tts.test_tts(text):
                    print(f"{Fore.GREEN}✅ TTS test successful{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ TTS test failed{Style.RESET_ALL}")
                    
    except Exception as e:
        print(f"{Fore.RED}❌ Error during voice test: {e}{Style.RESET_ALL}")


@main.command("clone-voice")
@click.argument("text")
@click.option("--reference", type=click.Path(exists=True), help="Reference audio file")
@click.option("--speaker", help="Use trained speaker model")
@click.option("--output", type=click.Path(), help="Output audio file path")
@click.option("--play", is_flag=True, help="Play the generated audio")
def clone_voice(text, reference, speaker, output, play):
    """Clone voice with specific text using reference audio or trained model."""
    
    print(f"{Fore.CYAN}🎭 Voice Cloning{Style.RESET_ALL}")
    print("=" * 50)
    print(f"Text: \"{text}\"")
    
    if reference:
        print(f"Reference: {reference}")
    elif speaker:
        print(f"Speaker model: {speaker}")
    else:
        print("Using default reference audio")
    
    print("")
    
    try:
        from .tts import TextToSpeech
        tts = TextToSpeech()
        
        if not tts.xtts_cloner:
            print(f"{Fore.RED}❌ XTTS not available. Voice cloning requires Python 3.10/3.11.{Style.RESET_ALL}")
            return
        
        # Load XTTS model if not loaded
        if not tts.xtts_cloner.model:
            print("🔄 Loading XTTS model...")
            if not tts.load_xtts_model():
                print(f"{Fore.RED}❌ Failed to load XTTS model{Style.RESET_ALL}")
                return
        
        result_path = None
        
        # Clone with reference audio
        if reference:
            result_path = tts.clone_voice_from_audio(text, reference, output)
        
        # Clone with speaker model
        elif speaker:
            result_path = tts.synthesize_with_speaker(text, speaker, output)
        
        # Clone with default reference
        else:
            if config.reference_audio_path.exists():
                result_path = tts.clone_voice_from_audio(text, str(config.reference_audio_path), output)
            else:
                print(f"{Fore.RED}❌ No default reference audio found{Style.RESET_ALL}")
                print(f"Record reference: soumyagpt setup-voice")
                return
        
        if result_path:
            print(f"{Fore.GREEN}✅ Voice cloning successful!{Style.RESET_ALL}")
            print(f"Generated: {result_path}")
            
            if play:
                print("🔊 Playing generated audio...")
                tts.play_audio(result_path)
        else:
            print(f"{Fore.RED}❌ Voice cloning failed{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error during voice cloning: {e}{Style.RESET_ALL}")


@main.command("devices")
def devices():
    """List available audio devices."""
    
    print(f"{Fore.CYAN}🎤 Audio Devices{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        recorder = VoiceRecorder()
        device_info = recorder.list_audio_devices()
        
        input_devices = device_info.get("input_devices", [])
        default_device = device_info.get("default_device")
        
        if input_devices:
            print(f"{Fore.GREEN}Input Devices:{Style.RESET_ALL}")
            for device in input_devices:
                device_id = device["id"]
                name = device["name"]
                channels = device["channels"]
                sample_rate = device["sample_rate"]
                
                default_indicator = " [DEFAULT]" if device_id == default_device else ""
                print(f"  {device_id}: {name}{default_indicator}")
                print(f"      Channels: {channels}, Sample Rate: {sample_rate}Hz")
        else:
            print(f"{Fore.YELLOW}No input devices found{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error listing devices: {e}{Style.RESET_ALL}")


# Update the existing setup-voice command to include voice cloning setup
@main.command("setup-voice")
@click.option("--duration", type=float, default=20.0, help="Recording duration in seconds")
def setup_voice_enhanced(duration):
    """Set up voice samples and test voice cloning (enhanced version)."""
    
    print(f"{Fore.CYAN}🎭 Enhanced Voice Setup{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        from .tts import TextToSpeech
        tts = TextToSpeech()
        
        # Show voice cloning setup
        tts.setup_voice_cloning()
        
        # Record reference voice
        if click.confirm("\nWould you like to record reference audio for voice cloning?"):
            success = tts.record_reference_voice(duration)
            if success:
                print(f"{Fore.GREEN}✅ Reference audio recording complete!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Reference audio recording failed{Style.RESET_ALL}")
        
        # Test voice synthesis
        if click.confirm("\nWould you like to test voice synthesis?"):
            tts.test_voice_clone()
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Voice setup cancelled{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error during voice setup: {e}{Style.RESET_ALL}")