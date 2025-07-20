import hashlib
import re
import logging
from pathlib import Path

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_folders(output_dir: Path):
    try:
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)
        (output_dir / "mixed_data").mkdir(parents=True, exist_ok=True)
        (output_dir / "video_archives").mkdir(parents=True, exist_ok=True)
        (output_dir / "lancedb").mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directories at {output_dir}")
    except Exception as e:
        logging.error(f"Error creating directories at {output_dir}: {str(e)}")
        raise

def get_video_id(video_url: str) -> str:
    return hashlib.md5(video_url.encode()).hexdigest()

def validate_youtube_url(url: str) -> bool:
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
    return bool(re.match(youtube_regex, url)) 