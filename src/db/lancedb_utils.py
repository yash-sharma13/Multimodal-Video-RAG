import logging
from pathlib import Path
from typing import List, Optional
import lancedb
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_vector_stores(lancedb_dir: Path):
    try:
        text_vector_store = LanceDBVectorStore(
            uri=str(lancedb_dir),
            table_name="text_collection"
        )
        image_vector_store = LanceDBVectorStore(
            uri=str(lancedb_dir),
            table_name="image_collection"
        )
        logging.info(f"Set up vector stores at {lancedb_dir}")
        return text_vector_store, image_vector_store
    except Exception as e:
        logging.error(f"Error setting up vector stores: {str(e)}")
        raise

def check_table_exists(lancedb_dir: Path, table_name: str) -> bool:
    try:
        db = lancedb.connect(str(lancedb_dir))
        return table_name in db.table_names()
    except Exception as e:
        logging.debug(f"Table {table_name} does not exist: {str(e)}")
        return False

def check_video_exists(video_id: str, lancedb_dir: Path) -> bool:
    try:
        db = lancedb.connect(str(lancedb_dir))
        if 'image_collection' in db.table_names():
            table = db.open_table("image_collection")
            df = table.to_pandas()
            exists = video_id in set(row['metadata']['video_id'] for _, row in df.iterrows())
            logging.info(f"Video {video_id} exists in database: {exists}")
            return exists
        return False
    except Exception as e:
        logging.error(f"Error checking video {video_id}: {str(e)}")
        return False

def verify_persistence(lancedb_dir: Path) -> List[str]:
    try:
        db = lancedb.connect(str(lancedb_dir))
        video_ids = []
        if 'image_collection' in db.table_names():
            table = db.open_table("image_collection")
            df = table.to_pandas()
            video_ids = list(set(row['metadata']['video_id'] for _, row in df.iterrows() if 'video_id' in row['metadata']))
            logging.info(f"Found {len(video_ids)} videos in LanceDB: {video_ids}")
        return video_ids
    except Exception as e:
        logging.error(f"Error verifying persistence: {str(e)}")
        raise

def update_image_paths_in_lancedb(lancedb_dir: Path, video_id: str, old_path_prefix: Path, new_path_prefix: Path):
    logging.info(f"Skipping image path update for video {video_id} as files remain in mixed_data")
    pass 