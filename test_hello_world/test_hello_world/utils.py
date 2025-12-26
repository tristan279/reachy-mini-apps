"""Utility functions for AWS Polly (TTS) and AWS Transcribe (STT)."""

import logging
import os
import sys
import tempfile
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Set up logging to both file and console (same as main.py)
LOG_FILE = "/tmp/test_hello_world.log"
# Only configure if not already configured (to avoid duplicate handlers)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)


def get_polly_client():
    """Get AWS Polly client with credentials from environment or AWS credentials file."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "eu-west-1")

    # If credentials provided via environment variables, use them explicitly
    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "polly",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
    
    # Otherwise, use boto3's default credential chain (checks ~/.aws/credentials, IAM roles, etc.)
    # This will work if credentials are configured via 'aws configure'
    try:
        return boto3.client("polly", region_name=aws_region)
    except Exception as e:
        error_msg = str(e)
        if "Unable to locate credentials" in error_msg or "NoCredentialsError" in error_msg:
            raise ValueError(
                "AWS credentials not found. Please configure credentials using:\n"
                "  1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                "  2. AWS credentials file: Run 'aws configure' or create ~/.aws/credentials\n"
                f"Original error: {e}"
            )
        # Re-raise other errors (network issues, permission errors, etc.)
        raise


def get_transcribe_client():
    """Get AWS Transcribe client with credentials from environment or AWS credentials file."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "eu-west-1")

    # If credentials provided via environment variables, use them explicitly
    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "transcribe",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
    
    # Otherwise, use boto3's default credential chain (checks ~/.aws/credentials, IAM roles, etc.)
    # This will work if credentials are configured via 'aws configure'
    try:
        return boto3.client("transcribe", region_name=aws_region)
    except Exception as e:
        error_msg = str(e)
        if "Unable to locate credentials" in error_msg or "NoCredentialsError" in error_msg:
            raise ValueError(
                "AWS credentials not found. Please configure credentials using:\n"
                "  1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                "  2. AWS credentials file: Run 'aws configure' or create ~/.aws/credentials\n"
                f"Original error: {e}"
            )
        # Re-raise other errors (network issues, permission errors, etc.)
        raise


def speak(
    text: str,
    voice_id: str = "Joanna",
    output_format: str = "mp3",
    engine: str = "standard",  # "standard" is cheaper than "neural"
    output_path: Optional[str] = None,
) -> str:
    """
    Convert text to speech using AWS Polly (using cheapest options by default).

    Args:
        text: Text to convert to speech
        voice_id: AWS Polly voice ID (default: "Joanna")
        output_format: Output audio format (mp3, ogg_vorbis, pcm) - default: mp3
        engine: TTS engine ("standard" or "neural") - default: standard (cheaper)
        output_path: Optional path to save audio file. If None, uses temp file.

    Returns:
        Path to the generated audio file

    Raises:
        ValueError: If AWS credentials are not configured
        BotoCoreError: If AWS API call fails
    """
    try:
        polly_client = get_polly_client()

        logger.info(f"Generating speech with AWS Polly: '{text[:50]}...'")

        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat=output_format,
            VoiceId=voice_id,
            Engine=engine,
        )

        # Save audio to file
        if output_path is None:
            # Create temporary file
            suffix = f".{output_format}" if output_format != "pcm" else ".wav"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            output_path = temp_file.name
            temp_file.close()

        # Write audio stream to file
        with open(output_path, "wb") as f:
            f.write(response["AudioStream"].read())

        logger.info(f"Speech saved to: {output_path}")
        return output_path

    except ValueError as e:
        logger.error(f"AWS credentials error: {e}")
        raise
    except (BotoCoreError, ClientError) as e:
        logger.error(f"AWS Polly error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in speak(): {e}")
        raise


def record(
    audio_file_path: str,
    language_code: str = "en-US",
    job_name: Optional[str] = None,
    wait_for_completion: bool = True,
) -> str:
    """
    Transcribe audio file using AWS Transcribe (using cheapest standard transcription).

    Args:
        audio_file_path: Path to audio file to transcribe
        language_code: Language code (default: "en-US")
        job_name: Optional custom job name. If None, generates from filename.
        wait_for_completion: If True, waits for transcription to complete (default: True)

    Returns:
        Transcribed text

    Note:
        Uses standard transcription (cheapest option). Audio file must be uploaded to S3 first.

    Raises:
        ValueError: If AWS credentials are not configured
        BotoCoreError: If AWS API call fails
        FileNotFoundError: If audio file doesn't exist
    """
    try:
        transcribe_client = get_transcribe_client()

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Generate job name if not provided
        if job_name is None:
            base_name = os.path.basename(audio_file_path)
            job_name = f"transcribe_{base_name}_{int(os.path.getmtime(audio_file_path))}"

        logger.info(f"Starting transcription job: {job_name}")

        # Upload audio file to S3 (Transcribe requires S3 URI)
        # For simplicity, we'll use a local file approach with Transcribe Streaming
        # But standard Transcribe requires S3, so we'll use a workaround
        # Note: For production, you should upload to S3 first

        # Get S3 bucket from env (optional)
        s3_bucket = os.getenv("AWS_S3_BUCKET")
        s3_key = os.getenv("AWS_S3_KEY_PREFIX", "transcribe/") + os.path.basename(audio_file_path)

        if s3_bucket:
            # Upload to S3 first
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "eu-west-1"),
            )
            s3_client.upload_file(audio_file_path, s3_bucket, s3_key)
            media_uri = f"s3://{s3_bucket}/{s3_key}"
            logger.info(f"Uploaded audio to S3: {media_uri}")
        else:
            # For local files, we need to use Transcribe Streaming API instead
            # This is a simplified version - for local files, consider using streaming
            raise ValueError(
                "AWS_S3_BUCKET not set. AWS Transcribe requires audio files to be in S3. "
                "Please set AWS_S3_BUCKET in .env file, or use the streaming API."
            )

        # Start transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": media_uri},
            MediaFormat=os.path.splitext(audio_file_path)[1][1:],  # Get extension without dot
            LanguageCode=language_code,
        )

        logger.info(f"Transcription job started: {job_name}")

        if wait_for_completion:
            # Wait for job to complete
            import time

            while True:
                response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]

                if status == "COMPLETED":
                    # Get transcription result
                    transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                    import urllib.request
                    import json

                    with urllib.request.urlopen(transcript_uri) as url:
                        transcript_data = json.loads(url.read().decode())

                    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]
                    logger.info(f"Transcription completed: {transcript_text[:50]}...")

                    # Clean up transcription job
                    try:
                        transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                    except Exception as e:
                        logger.warning(f"Could not delete transcription job: {e}")

                    return transcript_text

                elif status == "FAILED":
                    failure_reason = response["TranscriptionJob"].get("FailureReason", "Unknown error")
                    raise Exception(f"Transcription job failed: {failure_reason}")

                # Wait before checking again
                time.sleep(2)

        else:
            # Return job name for async processing
            return job_name

    except ValueError as e:
        logger.error(f"AWS credentials error: {e}")
        raise
    except (BotoCoreError, ClientError) as e:
        logger.error(f"AWS Transcribe error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in record(): {e}")
        raise


def record_streaming(
    audio_stream,
    language_code: str = "en-US",
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Transcribe audio stream in real-time using AWS Transcribe Streaming API.

    Args:
        audio_stream: Audio data stream (bytes or file-like object)
        language_code: Language code (default: "en-US")
        sample_rate: Audio sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)

    Returns:
        Transcribed text

    Note:
        This is a simplified version. For production, you may want to use
        the streaming API with proper event handling.
    """
    try:
        transcribe_client = get_transcribe_client()

        logger.info("Starting streaming transcription...")

        # Note: AWS Transcribe Streaming API requires WebSocket connection
        # This is a placeholder - full implementation would require websocket handling
        # For now, we'll use a workaround with temporary file

        # Save stream to temp file and use regular transcribe
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_stream if isinstance(audio_stream, bytes) else audio_stream.read())
            temp_path = temp_file.name

        try:
            return record(temp_path, language_code=language_code)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error in record_streaming(): {e}")
        raise

