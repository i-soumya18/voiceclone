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