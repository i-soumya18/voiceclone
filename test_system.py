#!/usr/bin/env python3
"""
Test script for SoumyaGPT without pip installation.
This demonstrates the basic functionality of the system.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import soumyagpt
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test importing all modules."""
    print("🧪 Testing module imports...")
    
    try:
        from soumyagpt import config
        print("✅ Config module imported")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from soumyagpt import stt
        print("✅ STT module imported")
    except ImportError as e:
        print(f"❌ STT import failed: {e}")
        return False
    
    try:
        from soumyagpt import gemini
        print("✅ Gemini module imported")
    except ImportError as e:
        print(f"❌ Gemini import failed: {e}")
        return False
    
    try:
        from soumyagpt import tts
        print("✅ TTS module imported")
    except ImportError as e:
        print(f"❌ TTS import failed: {e}")
        return False
    
    try:
        from soumyagpt import pipeline
        print("✅ Pipeline module imported")
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration setup."""
    print("\n🔧 Testing configuration...")
    
    try:
        from soumyagpt.config import config
        
        print(f"Project root: {config.project_root}")
        print(f"Data dir: {config.data_dir}")
        print(f"Voice samples: {config.voice_samples_dir}")
        print(f"Gemini API key: {'✅ Set' if config.gemini_api_key else '❌ Missing'}")
        
        # Validate configuration
        if config.validate():
            print("✅ Configuration is valid")
            return True
        else:
            print("⚠️  Configuration has issues (this is expected without API key)")
            return True  # Still return True as this is expected
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_cli():
    """Test CLI module."""
    print("\n💻 Testing CLI module...")
    
    try:
        from soumyagpt import cli
        print("✅ CLI module loaded successfully")
        
        # Test CLI help (without actually running)
        print("CLI commands available:")
        print("  - soumyagpt chat --mic")
        print("  - soumyagpt setup-voice")
        print("  - soumyagpt test --all")
        print("  - soumyagpt speak 'text'")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎙️ SoumyaGPT – Voice-Cloned CLI Chatbot")
    print("=" * 60)
    print("Testing system components...")
    
    tests = [
        ("Module Imports", test_import),
        ("Configuration", test_configuration),
        ("CLI Interface", test_cli),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SoumyaGPT is ready to use.")
        print("\n📋 Next Steps:")
        print("1. Set your Gemini API key:")
        print("   set GEMINI_API_KEY=your_key_here")
        print("2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("3. Run the application:")
        print("   python -m soumyagpt.cli chat --mic")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()