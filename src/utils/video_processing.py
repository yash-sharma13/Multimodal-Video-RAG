import logging
import asyncio
import re
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable, AgeRestrictedError
from faster_whisper import WhisperModel
from fastapi import HTTPException
import retrying
from PIL import Image

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retrying.retry(
    stop_max_attempt_number=3,
    wait_fixed=2000,
    retry_on_exception=lambda e: isinstance(e, (VideoUnavailable, ConnectionError))
)
def download_video(url: str, output_path: Path) -> Path:
    try:
        yt = YouTube(url)
        logging.info(f"Checking availability for video: {url}")
        yt.check_availability()
        logging.info(f"Video title: {yt.title}, streams available: {len(yt.streams)}")
        streams = yt.streams.filter(progressive=True, file_extension='mp4')
        if not streams:
            logging.warning("No progressive MP4 streams, trying adaptive streams")
            streams = yt.streams.filter(file_extension='mp4', adaptive=True)
        if not streams:
            logging.warning("No MP4 streams, trying all streams")
            streams = yt.streams.filter(file_extension='mp4')
        if not streams:
            raise ValueError(f"No suitable streams available for {url}. Available streams: {yt.streams}")
        video = streams.order_by('resolution').desc().first()
        logging.info(f"Selected stream: {video}")
        video_path = video.download(output_path / "videos")
        logging.info(f"Video downloaded to {video_path}")
        return Path(video_path)
    except AgeRestrictedError:
        logging.error(f"Video {url} is age-restricted")
        raise HTTPException(status_code=403, detail="Video is age-restricted and cannot be accessed")
    except VideoUnavailable as e:
        logging.error(f"Video {url} is unavailable: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error downloading video {url}: {str(e)}")
        raise

def video_to_images(video_path: Path, output_folder: Path):
    try:
        video = VideoFileClip(str(video_path))
        duration = video.duration
        target_fps = 0.2
        frame_count = min(int(duration * target_fps), 100)
        if frame_count == 0:
            raise ValueError(f"No frames to extract from {video_path}")
        logging.info(f"Extracting ~{frame_count} frames at {target_fps} FPS")
        for i, frame in enumerate(video.iter_frames(fps=target_fps)):
            if i >= frame_count:
                break
            frame_path = output_folder / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)
            logging.debug(f"Saved frame {i} to {frame_path}")
        logging.info(f"Extracted {frame_count} frames to {output_folder}")
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {str(e)}")
        raise
    finally:
        if 'video' in locals():
            video.close()

def video_to_audio(video_path: Path, output_audio_path: Path) -> bool:
    try:
        video = VideoFileClip(str(video_path))
        if video.audio is None:
            logging.warning(f"Video {video_path} has no audio track")
            return False
        video.audio.write_audiofile(str(output_audio_path))
        if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
            logging.warning(f"Audio file {output_audio_path} is empty or not created")
            return False
        logging.info(f"Audio extracted to {output_audio_path}, size={output_audio_path.stat().st_size} bytes")
        return True
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {str(e)}")
        return False
    finally:
        if 'video' in locals():
            video.close()
            if video.audio:
                video.audio.close()

def audio_to_text(audio_path: Path) -> str:
    try:
        model = WhisperModel("base", device="cpu")
        segments, info = model.transcribe(str(audio_path))
        transcript = "".join(segment.text for segment in segments)
        transcript = re.sub(r'\s+', ' ', transcript.strip())
        if not transcript:
            logging.warning(f"No transcript generated for {audio_path}")
        else:
            logging.info(f"Transcribed audio from {audio_path}, language: {info.language}, length={len(transcript)}")
        return transcript
    except Exception as e:
        logging.error(f"Error transcribing audio {audio_path}: {str(e)}")
        return ""

def save_transcript(transcript: str, output_path: Path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        logging.info(f"Transcript saved to {output_path}, length={len(transcript)}")
    except Exception as e:
        logging.error(f"Error saving transcript to {output_path}: {str(e)}")
        raise 