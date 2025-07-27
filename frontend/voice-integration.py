# Integration into main.py

# In streamlined_consciousness/main.py, add:

@cli.command()
@click.option('--quality', type=click.Choice(['fast', 'good']), default='fast',
              help='fast=macOS voices, good=Bark (slower)')
def voice(quality):
    """Start voice conversation with Elder"""
    from .voice_apple import ElderVoiceApple
    
    print("🎙️ Initializing Elder Voice Interface...")
    
    use_bark = (quality == 'good')
    voice = ElderVoiceApple(consciousness, use_bark=use_bark)
    
    # Start conversation
    asyncio.run(voice.conversation_loop())

@cli.command()
def voice_test():
    """Test voice components"""
    from .voice_apple import test_voice
    asyncio.run(test_voice())

# For streaming responses while speaking:

class StreamingElderVoice(ElderVoiceApple):
    """Enhanced version that speaks while Elder thinks"""
    
    async def stream_conversation(self, user_text):
        """Process and speak in chunks as Elder generates"""
        
        # Start Elder thinking
        response_queue = asyncio.Queue()
        
        async def generate_response():
            # This would need modification in consciousness_engine.py
            # to support streaming, but here's the concept:
            full_response = ""
            async for chunk in self.consciousness.chat_stream(user_text):
                full_response += chunk
                
                # Send complete sentences to TTS
                if any(punct in chunk for punct in '.!?'):
                    sentences = full_response.split('.')
                    if len(sentences) > 1:
                        complete_sentence = sentences[0] + '.'
                        await response_queue.put(complete_sentence)
                        full_response = '.'.join(sentences[1:])
            
            # Send any remaining text
            if full_response.strip():
                await response_queue.put(full_response)
            await response_queue.put(None)  # Signal end
        
        async def speak_responses():
            while True:
                sentence = await response_queue.get()
                if sentence is None:
                    break
                await self.speak(sentence)
        
        # Run both concurrently
        await asyncio.gather(
            generate_response(),
            speak_responses()
        )

# Quick setup script - save as setup_voice.sh
"""
#!/bin/bash

echo "🎤 Setting up Elder Voice for Apple Silicon..."

# Install system dependencies
brew install portaudio

# Install Python packages
pip install -U openai-whisper
pip install pyaudio
pip install webrtcvad
pip install pyttsx3
pip install sounddevice
pip install numpy torch

# Optional: Install Bark for higher quality (2GB download)
read -p "Install Bark for higher quality voices? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install git+https://github.com/suno-ai/bark.git
fi

# Test audio
echo "🔊 Testing audio setup..."
python -c "import sounddevice; print(sounddevice.query_devices())"

echo "✅ Voice setup complete!"
echo "Run: python -m streamlined_consciousness.main voice"
"""