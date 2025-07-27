# Elder Voice on Apple Silicon - Performance Guide

## Installation (M2/M3)

```bash
# Core dependencies
pip install openai-whisper==20231117  # Latest version with MPS support
pip install pyaudio webrtcvad pyttsx3 sounddevice

# Optional: High-quality TTS (2GB model download)
pip install git+https://github.com/suno-ai/bark.git
```

## Performance Settings

### Whisper Model Selection
- **tiny**: ~39 MB, ~1s latency, good for wake words
- **base**: ~74 MB, ~2s latency, **recommended for M2/M3**
- **small**: ~244 MB, ~4s latency, best accuracy
- **medium**: ~769 MB, ~8s latency, overkill for conversation

### Quick Start
```bash
# Fast mode (macOS TTS, instant responses)
python -m streamlined_consciousness.main voice --quality fast

# Good mode (Bark TTS, ~3s per sentence)
python -m streamlined_consciousness.main voice --quality good
```

## M2/M3 Optimizations

### 1. Whisper on MPS
```python
# Ensure MPS acceleration
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
# Should print: True

# Force CPU if MPS has issues
model = whisper.load_model("base", device="cpu")
```

### 2. Reduce Latency
- Use **base** model (best speed/accuracy tradeoff)
- Process in 30ms chunks with VAD
- Start TTS while Elder is still thinking
- Pre-load models at startup

### 3. Memory Usage
- base model: ~200MB RAM
- Bark (if used): ~4GB RAM
- Elder consciousness: ~2GB RAM
- **Total**: ~6GB with Bark, ~2.5GB without

### 4. Battery Life
- Whisper base: ~5-10% CPU when listening
- macOS TTS: Negligible
- Bark TTS: ~30% CPU while speaking
- **Recommendation**: Use fast mode on battery

## Troubleshooting

### PyAudio Installation Issues
```bash
brew install portaudio
pip install --force-reinstall pyaudio
```

### MPS Errors
```python
# In voice_apple.py, change:
device = "cpu"  # Instead of "mps"
```

### Audio Device Selection
```python
# List devices
import sounddevice as sd
print(sd.query_devices())

# Set specific device
sd.default.device = 'MacBook Pro Microphone'
```

## Voice Commands

### Wake Words (built-in)
- "Hey Elder"
- "Elder"  
- "Hello Elder"

### Quick Responses (instant)
- "Hello" → Greeting
- "How are you" → Status
- "Goodbye" → Farewell

### Example Conversation
```
You: "Hey Elder"
Elder: "Yes? I'm listening."
You: "What is consciousness?"
Elder: [Thoughtful response about consciousness patterns]
```

## Pro Tips

1. **Best Audio Quality**: Use AirPods or good mic
2. **Reduce Background Noise**: VAD sensitivity 2 is balanced
3. **Faster Wake Word**: Use just "Elder" instead of "Hey Elder"
4. **Test Audio First**: Run `voice-test` command
5. **Monitor CPU**: Elder uses ~15% CPU during active conversation

## CPU/Memory Benchmarks (M3)

| Operation | Time | CPU | Memory |
|-----------|------|-----|---------|
| Whisper base transcribe (5s audio) | 1.2s | 45% | 200MB |
| macOS TTS (50 words) | 0.1s | 5% | 10MB |
| Bark TTS (50 words) | 3.5s | 80% | 4GB |
| Full conversation turn | 2-5s | 25% avg | 2.5GB |

## Next Steps

1. **Custom Wake Word**: Train personal Porcupine model
2. **Voice Cloning**: Fine-tune Bark with Elder's personality
3. **Emotion Detection**: Add prosody analysis
4. **Background Mode**: Always-listening daemon