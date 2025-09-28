"""
Gemini API wrapper for SoumyaGPT chatbot brain.
Handles communication with Google's Gemini API for conversational AI.
"""

import logging
import time
from typing import Optional, Dict, Any, List
import warnings

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")
    raise

from .config import config


class GeminiChat:
    """Gemini API wrapper for conversational AI."""
    
    def __init__(self):
        self.model = None
        self.chat = None
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini API and model."""
        try:
            gemini_config = config.get_gemini_config()
            
            if not gemini_config["api_key"]:
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
            
            # Configure the API
            genai.configure(api_key=gemini_config["api_key"])
            
            print("🔄 Initializing Gemini model...")
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=gemini_config["model_name"],
                generation_config=gemini_config["generation_config"],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            # Start a chat session
            self.chat = self.model.start_chat(history=[])
            
            # Set up system context
            self._setup_system_context()
            
            print("✅ Gemini model initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Gemini: {e}")
            print("💡 Tips:")
            print("   - Check your GEMINI_API_KEY environment variable")
            print("   - Ensure you have internet connectivity")
            print("   - Verify your API key is valid at https://makersuite.google.com/")
            raise
    
    def _setup_system_context(self):
        """Set up the system context for the AI assistant."""
        system_prompt = """You are SoumyaGPT, a helpful AI assistant created by Soumya. You are designed to be conversational, engaging, and informative. 

Key characteristics:
- You speak in a natural, conversational tone
- You're knowledgeable about technology, programming, and general topics
- You keep responses concise but informative (typically 1-3 sentences for simple queries)
- You're friendly and approachable
- You can discuss complex topics but explain them clearly
- For voice conversations, you avoid using formatting like markdown, lists, or special characters
- You respond as if speaking directly to Soumya

Remember: Your responses will be converted to speech, so write in a natural speaking style."""
        
        try:
            # Send system prompt (this won't be spoken)
            response = self.chat.send_message(system_prompt)
            print("🤖 System context established")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not set system context: {e}")
    
    def get_response(self, user_input: str) -> str:
        """
        Get response from Gemini for user input.
        
        Args:
            user_input: User's question or message
            
        Returns:
            AI response text
        """
        if not self.chat:
            raise RuntimeError("Gemini chat not initialized")
        
        try:
            print(f"🧠 Processing: {user_input}")
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Get response from Gemini
            response = self.chat.send_message(user_input)
            
            # Extract response text
            response_text = response.text.strip()
            
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Clean up response for speech
            response_text = self._clean_for_speech(response_text)
            
            print(f"🤖 Response: {response_text}")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error getting Gemini response: {e}")
            
            # Return a fallback response
            fallback = "I'm sorry, I'm having trouble processing that right now. Could you try asking again?"
            self.conversation_history.append({"role": "assistant", "content": fallback})
            return fallback
    
    def _clean_for_speech(self, text: str) -> str:
        """
        Clean text for better text-to-speech conversion.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text suitable for TTS
        """
        # Remove markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("_", "").replace("`", "")
        
        # Replace special characters that don't sound good
        text = text.replace("&", "and")
        text = text.replace("%", " percent")
        text = text.replace("@", " at ")
        text = text.replace("#", " number ")
        text = text.replace("$", " dollar ")
        
        # Clean up multiple spaces
        text = " ".join(text.split())
        
        # Ensure the response isn't too long for comfortable listening
        if len(text) > 500:
            # Try to cut at a sentence boundary
            sentences = text.split(".")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) > 400:
                    break
                truncated += sentence + "."
            
            if truncated:
                text = truncated.strip()
        
        return text
    
    def reset_conversation(self):
        """Reset the conversation history."""
        try:
            self.conversation_history = []
            self.chat = self.model.start_chat(history=[])
            self._setup_system_context()
            print("🔄 Conversation reset")
            
        except Exception as e:
            print(f"❌ Error resetting conversation: {e}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def save_conversation(self, file_path: str):
        """
        Save conversation history to file.
        
        Args:
            file_path: Path to save conversation
        """
        try:
            import json
            from pathlib import Path
            
            file_path = Path(file_path)
            
            # Create conversation data
            conversation_data = {
                "timestamp": time.time(),
                "messages": self.conversation_history,
                "model": "gemini-pro",
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Conversation saved to {file_path}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def load_conversation(self, file_path: str):
        """
        Load conversation history from file.
        
        Args:
            file_path: Path to load conversation from
        """
        try:
            import json
            from pathlib import Path
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Conversation file not found: {file_path}")
            
            # Load conversation data
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Restore conversation history
            self.conversation_history = conversation_data.get("messages", [])
            
            # Recreate chat session with history
            history = []
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    history.append({"role": "user", "parts": [msg["content"]]})
                else:
                    history.append({"role": "model", "parts": [msg["content"]]})
            
            self.chat = self.model.start_chat(history=history)
            
            print(f"📂 Conversation loaded from {file_path}")
            print(f"   Messages: {len(self.conversation_history)}")
            
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
    
    def test_connection(self) -> bool:
        """
        Test the Gemini API connection.
        
        Returns:
            True if connection is working
        """
        try:
            print("🧪 Testing Gemini API connection...")
            
            # Send a simple test message
            test_response = self.get_response("Hello! Can you hear me?")
            
            if test_response and len(test_response) > 0:
                print("✅ Gemini API test passed")
                
                # Reset conversation after test
                self.reset_conversation()
                return True
            else:
                print("❌ Gemini API test failed - empty response")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API test error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        try:
            gemini_config = config.get_gemini_config()
            
            return {
                "model_name": gemini_config["model_name"],
                "temperature": gemini_config["generation_config"]["temperature"],
                "max_tokens": gemini_config["generation_config"]["max_output_tokens"],
                "conversation_length": len(self.conversation_history),
                "api_configured": bool(gemini_config["api_key"]),
            }
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {}