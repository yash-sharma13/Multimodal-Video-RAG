import logging
from pathlib import Path
from shutil import copytree, rmtree, make_archive, unpack_archive
import lancedb
from db.lancedb_utils import check_video_exists, verify_persistence

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def restore_video_data(video_id: str, mixed_data_dir: Path, archive_dir: Path):
    try:
        video_data_dir = mixed_data_dir / video_id
        archive_path = archive_dir / f"{video_id}.zip"
        if not video_data_dir.exists() and archive_path.exists():
            unpack_archive(archive_path, archive_dir, format='zip')
            # After unpacking, the content is in a folder named after video_id inside archive_dir
            # We need to move it to mixed_data_dir
            temp_unpacked_dir = archive_dir / video_id
            if temp_unpacked_dir.exists():
                copytree(temp_unpacked_dir, video_data_dir, dirs_exist_ok=True)
                rmtree(temp_unpacked_dir)
                logging.info(f"Restored video data for {video_id} from {archive_path} to {video_data_dir}")
            else:
                logging.warning(f"Unpacked directory {temp_unpacked_dir} not found for {video_id}")
        elif video_data_dir.exists():
            logging.debug(f"Video data for {video_id} already exists at {video_data_dir}")
        else:
            logging.warning(f"No archive found for {video_id} at {archive_path}")
    except Exception as e:
        logging.error(f"Error restoring video data for {video_id}: {str(e)}")

def clear_mixed_data(mixed_data_dir: Path, current_video_id: str, lancedb_dir: Path, archive_dir: Path):
    try:
        db = lancedb.connect(str(lancedb_dir))
        video_ids_in_db = set()
        if 'image_collection' in db.table_names():
            table = db.open_table("image_collection")
            df = table.to_pandas()
            video_ids_in_db = set(row['metadata']['video_id'] for _, row in df.iterrows())

        # Get list of video IDs that have archives
        archived_video_ids = set(f.stem for f in archive_dir.glob("*.zip"))

        for folder in mixed_data_dir.iterdir():
            if folder.is_dir() and folder.name != current_video_id:
                # Preserve if it exists in LanceDB or has an archive
                if folder.name in video_ids_in_db or folder.name in archived_video_ids:
                    logging.info(f"Preserving data for video {folder.name} as it exists in LanceDB or archive")
                    # No need to restore here, just ensure it's not deleted
                else:
                    rmtree(folder)
                    logging.info(f"Cleared stale data: {folder}")
    except Exception as e:
        logging.error(f"Error clearing mixed_data: {str(e)}")

def archive_video_data(video_id: str, mixed_data_dir: Path, archive_dir: Path, lancedb_dir: Path):
    try:
        video_data_dir = (mixed_data_dir / video_id).resolve()
        archive_path = (archive_dir / f"{video_id}").resolve() # make_archive expects a base name without .zip
        archive_zip = archive_dir / f"{video_id}.zip"

        if video_data_dir.exists():
            if archive_zip.exists():
                logging.info(f"Archive {archive_zip} already exists for {video_id}")
            else:
                make_archive(str(archive_path), 'zip', video_data_dir)
                logging.info(f"Created archive {archive_zip} for {video_id}")
        else:
            logging.warning(f"Video data directory {video_data_dir} does not exist for archiving")
    except Exception as e:
        logging.error(f"Error archiving video data: {str(e)}")
        raise 