# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-09-28

### Added
- Initial release of SoumyaGPT voice-cloned CLI chatbot
- Speech-to-Text using faster-whisper (GPU optimized)
- Google Gemini API integration for conversational AI
- Text-to-Speech with voice cloning using Coqui TTS
- Click-based CLI interface with comprehensive commands
- Configuration management with environment variables and JSON config
- Voice cloning setup and testing capabilities
- Continuous conversation mode
- Audio file processing support
- Conversation history management
- Performance metrics and monitoring
- Comprehensive testing suite
- Hardware-friendly optimization for GTX 1650 + i5 + 24GB RAM

### Features
- `soumyagpt chat --mic` - Voice conversation with microphone
- `soumyagpt chat --file` - Process audio files
- `soumyagpt setup-voice` - Set up voice cloning
- `soumyagpt test --all` - Test all components
- `soumyagpt speak "text"` - Text-to-speech conversion
- `soumyagpt history` - Manage conversation history

### Technical Details
- Supports Python 3.8+
- Modular architecture with separate STT, TTS, and LLM components
- Configurable model settings and audio parameters
- Error handling and fallback mechanisms
- Cross-platform compatibility (Windows, macOS, Linux)