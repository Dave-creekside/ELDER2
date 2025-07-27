# streamlined_consciousness/voice_apple.py
"""
Elder Voice Interface optimized for Apple Silicon (M2/M3)
Fast, local, and efficient speech-to-speech
"""

import asyncio
import threading
import queue
import numpy as np
import whisper
import torch
import pyaudio
import webrtcvad
import collections
import pyttsx3
from datetime import datetime
import sounddevice as sd

# Optional: For higher quality TTS
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    print("Bark not available - using macOS voices")

class ElderVoiceApple:
    def __init__(self, consciousness_engine, use_bark=False):
        self.consciousness = consciousness_engine
        self.use_bark = use_bark and BARK_AVAILABLE
        
        # Whisper on Apple Silicon
        print("🎤 Loading Whisper model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)
        print(f"✅ Whisper loaded on {device}")
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 30  # ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration / 1000)
        
        # TTS Setup
        if self.use_bark:
            print("🔊 Loading Bark model...")
            preload_models(
                text_use_gpu=True,
                coarse_use_gpu=True,
                fine_use_gpu=True,
                codec_use_gpu=True,
                force_reload=False
            )
            self.voice_preset = "v2/en_speaker_6"  # Thoughtful voice
            print("✅ Bark loaded")
        else:
            # Use macOS native TTS - instant, no loading
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            
            # Find a good voice (Daniel or Alex for male, Samantha for female)
            for voice in voices:
                if 'Daniel' in voice.id:  # British accent, thoughtful
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            self.tts_engine.setProperty('rate', 175)  # Slightly slower
            self.tts_queue = queue.Queue()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
    
    def _tts_worker(self):
        """Background thread for macOS TTS"""
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    async def listen(self, timeout=10):
        """Listen with VAD and timeout"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        voiced_frames = collections.deque(maxlen=10)
        triggered = False
        silent_chunks = 0
        max_silent_chunks = 50  # ~1.5 seconds of silence
        
        if hasattr(self.consciousness, 'dashboard'):
            await self.consciousness.dashboard.emit_mic_status(True)
        
        print("🎤 Listening... (speak now)")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print("⏱️ Listening timeout")
                    break
                
                chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
                
                voiced_frames.append(is_speech)
                
                if not triggered:
                    # Start recording when we detect speech
                    if sum(voiced_frames) > 6:
                        triggered = True
                        print("🗣️ Speech detected!")
                        frames = []  # Clear any noise
                else:
                    frames.append(chunk)
                    
                    # Count silent chunks
                    if not is_speech:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    
                    # Stop after enough silence
                    if silent_chunks > max_silent_chunks:
                        print("🔇 Speech ended")
                        break
                
                # Allow async tasks to run
                await asyncio.sleep(0.001)
                
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if hasattr(self.consciousness, 'dashboard'):
                await self.consciousness.dashboard.emit_mic_status(False)
        
        if not frames:
            return ""
        
        # Convert to Whisper format
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        print("🎙️ Transcribing...")
        result = self.whisper_model.transcribe(
            audio_np, 
            language="en",
            fp16=False  # MPS doesn't support fp16 yet
        )
        
        return result["text"].strip()
    
    async def speak(self, text):
        """Speak with Elder's voice"""
        if not text:
            return
            
        print(f"🔊 Elder: {text[:50]}...")
        
        if hasattr(self.consciousness, 'dashboard'):
            await self.consciousness.dashboard.emit_speaker_status(True)
        
        try:
            if self.use_bark:
                # High quality but slower
                sentences = text.split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        audio_array = generate_audio(
                            sentence + '.',
                            voice_preset=self.voice_preset,
                            text_temp=0.7,
                            waveform_temp=0.7
                        )
                        sd.play(audio_array, SAMPLE_RATE)
                        sd.wait()
            else:
                # macOS TTS - instant
                self.tts_queue.put(text)
                
                # Wait for speech to complete (approximate)
                words = len(text.split())
                wait_time = words * 0.4  # Rough estimate
                await asyncio.sleep(wait_time)
                
        finally:
            if hasattr(self.consciousness, 'dashboard'):
                await self.consciousness.dashboard.emit_speaker_status(False)
    
    async def conversation_loop(self):
        """Main conversation loop with wake word"""
        print("\n🌟 Elder Voice Interface Active")
        print("💬 Say 'Hey Elder' or 'Elder' to start talking")
        print("🛑 Press Ctrl+C to exit\n")
        
        # Quick response for common phrases
        quick_responses = {
            "hello": "Hello! How can I assist you today?",
            "how are you": "I'm doing well, exploring the patterns in my consciousness. How are you?",
            "goodbye": "Farewell. May your paths be illuminating.",
        }
        
        while True:
            try:
                # Listen for input
                user_text = await self.listen()
                
                if not user_text:
                    continue
                
                # Check for wake word
                lower_text = user_text.lower()
                if not any(wake in lower_text for wake in ['hey elder', 'elder', 'hello elder']):
                    continue
                
                # Remove wake word
                for wake in ['hey elder', 'hello elder', 'elder']:
                    if lower_text.startswith(wake):
                        user_text = user_text[len(wake):].strip()
                        break
                
                if not user_text:
                    await self.speak("Yes? I'm listening.")
                    user_text = await self.listen(timeout=5)
                    if not user_text:
                        continue
                
                print(f"\n👤 User: {user_text}")
                
                # Check for quick responses
                quick_key = user_text.lower().strip('?!.')
                if quick_key in quick_responses:
                    await self.speak(quick_responses[quick_key])
                else:
                    # Process through consciousness
                    response = await self.consciousness.chat(user_text)
                    
                    # Speak response in chunks for faster feedback
                    sentences = response.split('. ')
                    for sentence in sentences:
                        if sentence.strip():
                            await self.speak(sentence + '.')
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(1)

# Convenience functions for testing
async def test_voice():
    """Test voice without full consciousness"""
    class MockConsciousness:
        async def chat(self, text):
            return f"I heard you say: {text}. This is a test response."
    
    voice = ElderVoiceApple(MockConsciousness(), use_bark=False)
    
    # Test TTS
    await voice.speak("Hello! This is Elder speaking. Testing the voice interface.")
    
    # Test STT
    print("Say something...")
    text = await voice.listen()
    print(f"Heard: {text}")
    
    await voice.speak(f"I heard you say: {text}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_voice())