# Cost Optimization Guide

## AWS Transcribe Pricing Model

### How AWS Charges

**Streaming Transcription:**
- **$0.0275 per minute** of audio sent to AWS
- Charges are based on **audio duration sent**, not connection time
- Minimum charge: 15 seconds per request
- Billed in 1-second increments

**Key Insight:** AWS charges for **audio data sent**, not for the WebSocket connection time. This means:
- ✅ If you don't send audio during silence, you don't pay for silence
- ❌ If you continuously stream (including silence), you pay for everything

### Cost Comparison

**Scenario: 1 hour conversation per day**

**Without VAD (Continuous Streaming):**
- Total audio streamed: 60 minutes/day
- Monthly cost: 60 min × 30 days × $0.0275 = **$49.50/month**

**With VAD (Speech-Only Streaming):**
- Assumes 30% of time is actual speech (typical conversation)
- Audio streamed: 18 minutes/day (30% of 60 min)
- Monthly cost: 18 min × 30 days × $0.0275 = **$14.85/month**
- **Savings: 70% reduction ($34.65/month)**

**With VAD (20% speech ratio - more realistic):**
- Audio streamed: 12 minutes/day
- Monthly cost: 12 min × 30 days × $0.0275 = **$9.90/month**
- **Savings: 80% reduction ($39.60/month)**

## How VAD Works

### Simple VAD (Default)
- Uses energy-based detection (RMS)
- Detects when audio energy exceeds threshold
- Lightweight, no extra dependencies
- Good for most use cases

### WebRTC VAD (Recommended)
- More accurate speech detection
- Better at filtering out background noise
- Requires: `pip install webrtcvad`
- Recommended for noisy environments

### Implementation Details

1. **Audio Buffering**: When not speaking, audio is buffered (last 500ms)
2. **Speech Detection**: When speech starts, buffered audio + current audio is sent
3. **Transmission**: Only audio during speech is sent to AWS
4. **Connection**: WebSocket stays open, but no charges for silence

## Configuration

### Enable VAD (Default)

VAD is enabled by default. To explicitly enable:

```bash
export USE_VAD=true
export VAD_TYPE=simple  # or "webrtc"
```

### Disable VAD (Not Recommended)

Only disable if you need continuous transcription:

```bash
export USE_VAD=false
```

### WebRTC VAD Setup

For better accuracy:

```bash
pip install webrtcvad
export VAD_TYPE=webrtc
```

## Cost Savings Examples

### Example 1: Home Assistant (Low Usage)
- Usage: 30 minutes/day, 20% speech
- Without VAD: $24.75/month
- With VAD: $4.95/month
- **Savings: $19.80/month (80%)**

### Example 2: Office Assistant (Medium Usage)
- Usage: 2 hours/day, 25% speech
- Without VAD: $99/month
- With VAD: $24.75/month
- **Savings: $74.25/month (75%)**

### Example 3: Customer Service Bot (High Usage)
- Usage: 8 hours/day, 30% speech
- Without VAD: $396/month
- With VAD: $118.80/month
- **Savings: $277.20/month (70%)**

## Best Practices

1. **Always use VAD** - No reason not to (enabled by default)
2. **Use WebRTC VAD** in noisy environments
3. **Tune VAD thresholds** if needed (see `vad.py`)
4. **Monitor costs** in AWS Console
5. **Consider batch transcription** for non-real-time use cases

## Batch Transcription Alternative

For non-real-time use cases, batch transcription is cheaper:

- **Batch**: $0.024/minute (14% cheaper than streaming)
- **Trade-off**: Higher latency (seconds to minutes)
- **Use case**: Record conversations, process later

To use batch transcription, you'd need to:
1. Record audio locally
2. Detect speech segments
3. Send only speech segments to AWS Transcribe (batch API)
4. Process results

This is not implemented in the current codebase but could be added.

## Monitoring Costs

### AWS Cost Explorer
1. Go to AWS Console → Cost Explorer
2. Filter by service: "Amazon Transcribe"
3. View daily/monthly costs
4. Set up billing alerts

### Cost Alerts
```bash
# Set up CloudWatch billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name transcribe-cost-alert \
  --alarm-description "Alert when Transcribe costs exceed $50/month" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

## FAQ

**Q: Does VAD affect transcription accuracy?**
A: No. VAD only controls when audio is sent. AWS Transcribe accuracy is the same.

**Q: What if VAD misses speech?**
A: The buffer (500ms) helps capture speech starts. WebRTC VAD is more accurate.

**Q: Can I use batch transcription for real-time?**
A: No. Batch has 5-15 minute latency. Use streaming for real-time.

**Q: How much can I save?**
A: Typically 60-80% depending on speech ratio. More silence = more savings.

**Q: Does this work with AWS Transcribe's built-in VAD?**
A: AWS has server-side VAD, but you still pay for all audio sent. Client-side VAD prevents sending silence, saving money.

## Summary

✅ **VAD is enabled by default** - saves 60-80% on transcription costs
✅ **No accuracy loss** - only controls when audio is sent
✅ **Easy to configure** - just set environment variables
✅ **WebRTC VAD recommended** - better accuracy in noisy environments

**Bottom line:** Using VAD can reduce your AWS Transcribe costs by 60-80% with no downside!
