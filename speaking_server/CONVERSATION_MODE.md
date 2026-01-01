# Conversation Mode - AWS Transcribe + Polly

The speaking server now supports real-time voice conversation mode using AWS Transcribe for speech-to-text and AWS Polly for text-to-speech.

## Architecture

```
Robot Microphone → LocalStream → AWS Transcribe Streaming → LLM → AWS Polly → Robot Speaker
```

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

Or install manually:
```bash
pip install boto3 numpy scipy aiohttp amazon-transcribe
```

### 2. Configure AWS Credentials

Set up AWS credentials using one of these methods:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

**Option B: AWS CLI**
```bash
aws configure
```

**Option C: IAM Role** (if running on EC2)

### 3. Configure Conversation Mode

Set environment variables to enable conversation mode:

```bash
# Enable conversation mode
export CONVERSATION_MODE=true

# AWS Configuration
export AWS_REGION=us-east-1
export TRANSCRIBE_LANGUAGE=en-US
export POLLY_VOICE=Joanna
export POLLY_ENGINE=neural  # or "standard"

# LLM Configuration (optional)
export LLM_ENDPOINT=https://your-llm-endpoint.com/chat
export LLM_API_KEY=your-api-key  # if required
```

### 4. Run the App

```bash
# On the robot
speaking_server
```

Or if running directly:
```bash
python -m speaking_server.main
```

## Configuration Options

### AWS Transcribe
- **TRANSCRIBE_LANGUAGE**: Language code (default: `en-US`)
  - Supported: `en-US`, `en-GB`, `es-US`, `fr-FR`, `de-DE`, etc.
- **AWS_REGION**: AWS region (default: `us-east-1`)

### AWS Polly
- **POLLY_VOICE**: Voice ID (default: `Joanna`)
  - Popular voices: `Joanna`, `Matthew`, `Amy`, `Brian`, `Emma`, `Ivy`
  - Neural voices: `Joanna-Neural`, `Matthew-Neural`, `Amy-Neural`
- **POLLY_ENGINE**: Engine type (default: `neural`)
  - `neural`: Higher quality, more natural
  - `standard`: Lower cost, faster

### LLM Endpoint
- **LLM_ENDPOINT**: URL to your LLM service
  - If not set, uses simple echo mode ("You said: ...")
  - Expected request format: `POST {"text": "user message"}`
  - Expected response format: `{"response": "llm response"}` or `{"text": "..."}`

Example LLM endpoint implementations:
- AWS Bedrock
- OpenAI API
- Anthropic Claude API
- Custom LLM service

## Audio Specifications

### Input (Microphone → AWS Transcribe)
- **Format**: PCM, int16
- **Sample Rate**: 16,000 Hz (automatically resampled from robot's native rate)
- **Channels**: Mono (automatically converted from stereo if needed)

### Output (AWS Polly → Speaker)
- **Format**: PCM, int16 → float32
- **Sample Rate**: 24,000 Hz (automatically resampled to robot's native rate)
- **Channels**: Mono

## How It Works

1. **Audio Capture**: Robot's microphone captures audio continuously
2. **Speech-to-Text**: Audio is streamed to AWS Transcribe Streaming API
3. **Transcription**: AWS Transcribe returns partial and final transcripts
4. **LLM Processing**: Final transcripts are sent to your LLM endpoint
5. **Text-to-Speech**: LLM responses are synthesized using AWS Polly
6. **Audio Playback**: Generated audio is streamed to robot's speaker

## Troubleshooting

### "Error initializing AWS clients"
- Check AWS credentials are configured
- Verify AWS_REGION is correct
- Check network connectivity to AWS

### "amazon-transcribe not installed"
- Install: `pip install amazon-transcribe`
- The app will still work with boto3, but amazon-transcribe provides better async support

### No audio input
- Check robot's microphone is working: `robot.media.get_audio_sample()`
- Verify robot's media backend is configured
- Check sample rate: `robot.media.get_input_audio_samplerate()`

### No audio output
- Check robot's speaker is working
- Verify audio format conversion
- Check sample rate: `robot.media.get_output_audio_samplerate()`

### LLM endpoint errors
- Verify LLM_ENDPOINT URL is correct
- Check LLM_API_KEY if required
- Test endpoint with curl:
  ```bash
  curl -X POST https://your-llm-endpoint.com/chat \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello"}'
  ```

## Latency Considerations

- **STT Latency**: ~200-500ms (depends on network and AWS region)
- **LLM Latency**: Depends on your LLM service
- **TTS Latency**: ~300-800ms (AWS Polly processes complete text)
- **Total Latency**: ~1-2 seconds end-to-end

To reduce latency:
- Use `neural` engine for better quality (slightly faster)
- Use a closer AWS region
- Optimize LLM endpoint response time
- Consider chunking longer responses

## Cost Considerations

### AWS Transcribe Pricing

**Streaming Transcription:**
- **$0.0275 per minute** of audio streamed
- Charges for ALL audio, including silence
- Minimum charge: 15 seconds per request

**Batch Transcription (Alternative):**
- **$0.024 per minute** (first 250k minutes/month)
- Only charges for audio sent (not silence)
- Requires recording first, then sending

**Cost Optimization with VAD:**
- Use Voice Activity Detection (VAD) to only stream when speech is detected
- Can save **30-70%** on transcription costs (depending on silence ratio)
- Example: If 50% of time is silence, you save ~50% on costs

### AWS Polly
- Neural voices: ~$0.016 per 1000 characters
- Standard voices: ~$0.004 per 1000 characters

### Example Monthly Costs

**Without VAD (continuous streaming):**
- 1 hour/day of conversation (including silence):
  - Transcribe: ~$50/month (60 min/day × 30 days × $0.0275)
  - Polly (neural): ~$5-10/month
  - **Total: ~$55-60/month**

**With VAD (speech-only streaming):**
- 1 hour/day, but only 20 minutes of actual speech:
  - Transcribe: ~$16.50/month (20 min/day × 30 days × $0.0275)
  - Polly (neural): ~$5-10/month
  - **Total: ~$21-26/month**
  - **Savings: ~60% reduction**

### Enabling VAD

VAD is enabled by default. To configure:

```bash
export USE_VAD=true  # Enable VAD (default)
export VAD_TYPE=simple  # or "webrtc" for better accuracy
```

For better accuracy, install WebRTC VAD:
```bash
pip install webrtcvad
```

Then use:
```bash
export VAD_TYPE=webrtc
```

## Switching Between Modes

**HTTP Server Mode** (default):
```bash
unset CONVERSATION_MODE
# or
export CONVERSATION_MODE=false
```

**Conversation Mode**:
```bash
export CONVERSATION_MODE=true
```

## Example LLM Endpoint

Simple Flask/FastAPI endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(request: ChatRequest):
    # Your LLM logic here
    response = your_llm_function(request.text)
    return {"response": response}
```

## Next Steps

- Add voice activity detection (VAD) for better turn-taking
- Implement conversation history/context
- Add emotion detection and head movement synchronization
- Optimize for lower latency
- Add support for multiple languages
