# 🎙️ SoumyaGPT – Voice-Cloned CLI Chatbot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-yellow.svg)](https://github.com/yourusername/soumyagpt)

> **Your AI twin that speaks in your voice using Google's Gemini API**

SoumyaGPT is a standalone CLI chatbot that understands your voice, processes it through Google's Gemini API, and responds back in your cloned voice. Perfect for modest hardware setups (GTX 1650 + i5 + 24GB RAM).

## 🔹 Vision

Build a self-contained package that you can install via pip, run from the terminal, and instantly start a conversation with your AI twin that talks back like you.

## 🔹 Core Workflow

```
🎤 You speak → 🧠 Whisper STT → 🤖 Gemini API → 🗣️ Coqui TTS → 🔊 Your cloned voice
```

The system mimics human conversation:
1. **🎤 You speak** → speech-to-text
2. **🧠 Gemini processes** your query, generates response  
3. **🗣️ Response converted** back into your cloned voice
4. **🔊 Audio is played** in real-time

## ✨ Features

- ✅ **Voice input** via microphone or pre-recorded files
- ✅ **Real-time responses** from Gemini API
- ✅ **Voice cloning** so the chatbot replies in your voice
- ✅ **Installable package** (`pip install .`)
- ✅ **CLI-first design** → no heavy frontend required
- ✅ **Hardware-friendly** → optimized for GTX 1650 + i5
- ✅ **Continuous conversation** mode
- ✅ **Audio file processing**
- ✅ **Conversation history** management

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **STT** | `faster-whisper` | GPU-optimized speech transcription |
| **LLM Brain** | `Gemini API` | Offloads reasoning, no local compute |
| **TTS** | `Coqui TTS` | Voice cloning capabilities |
| **CLI** | `Click` | Command-line interface |
| **Audio** | `sounddevice + soundfile` | Mic capture & playback |

## 📋 Requirements

### Hardware
- **GPU**: GTX 1650 or better (CPU fallback available)
- **RAM**: 8GB+ (24GB recommended for best performance)
- **CPU**: Intel i5 or equivalent
- **Storage**: 2GB+ for models

### Software
- **Python**: 3.8 or higher
- **OS**: Windows, macOS, Linux
- **Microphone**: For voice input
- **Speakers/Headphones**: For audio output

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/soumyagpt.git
cd soumyagpt
```

### 2. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 3. Set Up API Keys

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your_gemini_api_key_here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your_gemini_api_key_here

# macOS/Linux
export GEMINI_API_KEY="your_gemini_api_key_here"
```

> **Get your Gemini API key**: Visit [Google AI Studio](https://makersuite.google.com/) to get your free API key.

### 4. Set Up Voice Cloning (Optional)

```bash
# Record your reference voice (20 seconds recommended)
soumyagpt setup-voice --duration 20

# Or manually place a WAV file at:
# data/voice_samples/reference.wav
```

### 5. Test Installation

```bash
# Test all components
soumyagpt test --all

# Test individual components
soumyagpt test --stt --tts --gemini --voice-clone
```

## 📖 Usage

### Basic Commands

```bash
# Start a voice conversation
soumyagpt chat --mic

# Process an audio file
soumyagpt chat --file input.wav

# Continuous conversation mode
soumyagpt chat --mic --continuous

# Text-only mode (no voice output)
soumyagpt chat --mic --no-voice

# Custom recording duration
soumyagpt chat --mic --duration 10

# Save generated audio
soumyagpt chat --mic --save-audio
```

### Advanced Usage

```bash
# Speak any text with your cloned voice
soumyagpt speak "Hello, this is my cloned voice!"

# Check system status
soumyagpt --status

# View configuration
soumyagpt --config

# Manage conversation history
soumyagpt history --save conversation.json
soumyagpt history --load conversation.json
soumyagpt history --reset
```

### Example Interaction

```bash
$ soumyagpt chat --mic

🎙️  SoumyaGPT – Voice-Cloned CLI Chatbot
═══════════════════════════════════════════════════════════════
Your AI twin that speaks in your voice using Gemini API

🎤 Recording for 5s... (speak now!)
✅ Recording completed
🧠 Transcribing audio...
📝 Text: What is deep learning?

🧠 Processing: What is deep learning?
🤖 SoumyaGPT: Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns in data. It's particularly powerful for tasks like image recognition, natural language processing, and speech synthesis.

🗣️  Converting to speech...
🎭 Using voice cloning...
🔊 Voice response played

⏱️  Total processing time: 3.45s
```

## 🏗️ Project Structure

```
soumyagpt/
├── soumyagpt/
│   ├── __init__.py      # Package initialization
│   ├── cli.py           # CLI commands and interface
│   ├── pipeline.py      # Main orchestration pipeline
│   ├── stt.py          # Speech-to-text (Whisper)
│   ├── tts.py          # Text-to-speech (Coqui TTS)
│   ├── gemini.py       # Gemini API wrapper
│   └── config.py       # Configuration management
├── data/
│   └── voice_samples/   # Your voice recordings
├── setup.py            # Package installation
├── pyproject.toml      # Modern Python packaging
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |

### Configuration File

Create `config.json` in the project root to customize settings:

```json
{
  "whisper_model": "base",
  "whisper_device": "auto",
  "sample_rate": 22050,
  "default_duration": 5.0,
  "temperature": 0.7,
  "max_response_tokens": 1000,
  "verbose": false
}
```

### Voice Cloning Setup

1. **Record Reference Audio**:
   ```bash
   soumyagpt setup-voice --duration 20
   ```

2. **Manual Setup**:
   - Record 10-30 seconds of clear speech
   - Save as `data/voice_samples/reference.wav`
   - Use 22050 Hz sample rate, mono channel

3. **Tips for Best Results**:
   - Use a quiet environment
   - Speak naturally and clearly  
   - Include varied intonation
   - Avoid background noise

## 🧪 Testing

### Component Testing

```bash
# Test speech-to-text
soumyagpt test --stt

# Test text-to-speech
soumyagpt test --tts

# Test Gemini API
soumyagpt test --gemini

# Test voice cloning
soumyagpt test --voice-clone

# Test everything
soumyagpt test --all
```

### Manual Testing

```bash
# Quick microphone test
soumyagpt chat --mic --duration 3

# Test with sample audio
soumyagpt chat --file path/to/audio.wav

# Test text-to-speech only
soumyagpt speak "Testing my voice clone"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "GEMINI_API_KEY not found"
```bash
# Set the environment variable
export GEMINI_API_KEY="your_key_here"

# Or add to ~/.bashrc (Linux/macOS)
echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc
```

#### 2. "No audio detected from microphone"
```bash
# Test microphone
soumyagpt test --stt

# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

#### 3. "TTS model failed to load"
```bash
# Try CPU mode in config.json
{
  "whisper_device": "cpu"
}

# Or use a smaller Whisper model
{
  "whisper_model": "tiny"
}
```

#### 4. "Voice cloning not working"
```bash
# Check reference audio exists
ls -la data/voice_samples/reference.wav

# Re-record reference voice
soumyagpt setup-voice --duration 20
```

### Performance Issues

#### Slow Response Times
- Use smaller Whisper model (`tiny` or `base`)
- Enable GPU acceleration
- Reduce `max_response_tokens` in config

#### High Memory Usage  
- Use CPU mode for Whisper
- Close other applications
- Restart the application periodically

#### Audio Quality Issues
- Check microphone quality
- Ensure quiet environment
- Adjust `sample_rate` in config

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/soumyagpt.git
cd soumyagpt
pip install -e .[dev]

# Run tests
pytest

# Format code
black soumyagpt/
flake8 soumyagpt/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for conversational AI
- **OpenAI Whisper** team for speech recognition
- **Coqui TTS** for voice synthesis
- **Click** for CLI framework

## 🔮 Roadmap

- [ ] **Multi-language support** for non-English conversations
- [ ] **Real-time streaming** for faster response times
- [ ] **Custom wake words** for hands-free activation
- [ ] **Emotion detection** and response adaptation
- [ ] **Voice activity detection** for automatic recording
- [ ] **Plugin system** for extending functionality
- [ ] **Desktop GUI** for visual interaction
- [ ] **Mobile app** companion

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/soumyagpt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/soumyagpt/discussions)
- **Email**: your-email@example.com

---

**Made with ❤️ by Soumya** | *Bringing AI voices to life*