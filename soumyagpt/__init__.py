"""
SoumyaGPT - Voice-Cloned CLI Chatbot

A standalone CLI chatbot that understands your voice, thinks using Google's Gemini API,
and responds in your cloned voice.
"""

__version__ = "0.1.0"
__author__ = "Soumya"
__email__ = "your-email@example.com"

from .cli import main
from .pipeline import VoiceChatPipeline

__all__ = ["main", "VoiceChatPipeline"]