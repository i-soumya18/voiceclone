# Python 3.13 Compatibility Notes

## Overview

SoumyaGPT has been updated to work with Python 3.13! However, some changes were made to ensure compatibility.

## Changes Made

### TTS Engine Updates
- **Replaced**: Coqui TTS (not compatible with Python 3.13)
- **Added**: Edge TTS (Microsoft's high-quality online TTS)
- **Added**: pyttsx3 (offline TTS fallback)

### Voice Cloning
- **Current**: Uses high-quality Edge TTS voices (not custom voice cloning)
- **Future**: Custom voice cloning available with Python 3.11 + Coqui TTS

## Installation (Python 3.13)

```powershell
# Install updated dependencies
pip install -r requirements.txt

# Or install package
pip install -e .
```

## Available TTS Options

### 1. Edge TTS (Recommended)
- **Quality**: Excellent, human-like voices
- **Internet**: Required
- **Voices**: Multiple high-quality options
- **Usage**: `soumyagpt speak "Hello world!"`

### 2. pyttsx3 (Offline Fallback) 
- **Quality**: Good, system voices
- **Internet**: Not required
- **Voices**: System-dependent
- **Usage**: Automatic fallback if Edge TTS fails

## Voice Options

Available Edge TTS voices:
- `en-US-AriaNeural` - Female, natural
- `en-US-GuyNeural` - Male, natural  
- `en-US-JennyNeural` - Female, friendly
- `en-US-DavisNeural` - Male, professional

## Migration from Coqui TTS

If you need custom voice cloning:

### Option 1: Use Python 3.11 Environment
```powershell
# Create Python 3.11 environment
conda create -n soumyagpt-clone python=3.11
conda activate soumyagpt-clone

# Install original dependencies with Coqui TTS
pip install TTS>=0.20.0
# ... other dependencies
```

### Option 2: Hybrid Setup
- Use Python 3.13 for main functionality
- Use Python 3.11 environment for voice cloning
- Share audio files between environments

## Testing

```powershell
# Test TTS system
soumyagpt test --tts

# Test all components  
soumyagpt test --all

# Speak with different voice
soumyagpt speak "Testing Edge TTS voice"
```

## Benefits of Python 3.13 + Edge TTS

✅ **Latest Python features** and performance improvements
✅ **High-quality voices** without local model downloads
✅ **Faster startup** - no heavy model loading
✅ **Multiple languages** supported by Edge TTS
✅ **No GPU required** for TTS processing
✅ **Reliable fallbacks** with pyttsx3

## Future Plans

- Custom voice cloning when Coqui TTS supports Python 3.13
- Additional TTS engine integrations
- Voice style transfer options
- Real-time voice modulation

---

**The core SoumyaGPT experience remains the same** - you still get intelligent conversations with high-quality voice responses! 🎙️✨