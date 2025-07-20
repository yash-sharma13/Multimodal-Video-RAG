import logging
import json
from pathlib import Path
from pytubefix import YouTube
from llama_index.core import StorageContext
from db.lancedb_utils import setup_vector_stores, check_video_exists, verify_persistence
from utils.common_utils import create_folders, validate_youtube_url, get_video_id
from utils.video_processing import download_video, video_to_images, video_to_audio, audio_to_text, save_transcript
from rag.index_management import load_documents, build_multimodal_index, load_existing_index
from rag.llm_models import GoogleGenAIMultiModal
from rag.retriever_response import setup_retriever, generate_response, plot_images
from data_management.archive_utils import archive_video_data, restore_video_data, clear_mixed_data
from constants import OUTPUT_DIR, MIXED_DATA_DIR, ARCHIVE_DIR, LANCEDB_DIR

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        output_dir = OUTPUT_DIR
        mixed_data_dir = MIXED_DATA_DIR
        archive_dir = ARCHIVE_DIR
        lancedb_dir = LANCEDB_DIR

        create_folders(output_dir)
        
        video_url = input("Enter the YouTube video URL: ").strip()
        if not video_url or not validate_youtube_url(video_url):
            raise ValueError("Invalid or empty YouTube URL. Please provide a valid YouTube link (e.g., https://www.youtube.com/watch?v=...). ")

        video_id = get_video_id(video_url)
        logging.info(f"Processing video with ID: {video_id}")

        logging.info("Verifying persistence of existing videos")
        verify_persistence(lancedb_dir)

        logging.info(f"Clearing stale data from {mixed_data_dir}")
        clear_mixed_data(mixed_data_dir, video_id, lancedb_dir, archive_dir)

        logging.info("Setting up vector stores")
        text_vector_store, image_vector_store = setup_vector_stores(lancedb_dir)
        storage_context = StorageContext.from_defaults(vector_store=text_vector_store)

        has_audio = True
        yt = YouTube(video_url)
        metadata = {
            "title": yt.title or "Unknown",
            "author": yt.author or "Unknown",
            "length": yt.length or 0,
            "publish_date": str(yt.publish_date) if yt.publish_date else "Unknown"
        }

        index = load_existing_index(lancedb_dir, storage_context, image_vector_store)
        video_ids = verify_persistence(lancedb_dir)
        documents = []

        if check_video_exists(video_id, lancedb_dir):
            logging.info(f"Video {video_id} already processed")
            restore_video_data(video_id, mixed_data_dir, archive_dir)
        else:
            logging.info("Starting video download")
            video_path = download_video(video_url, output_dir)
            video_data_dir = mixed_data_dir / video_id
            video_data_dir.mkdir(parents=True, exist_ok=True)

            logging.info("Starting frame extraction")
            video_to_images(video_path, video_data_dir)

            audio_path = video_data_dir / "audio.wav"
            logging.info("Starting audio extraction")
            has_audio = video_to_audio(video_path, audio_path)

            transcript = ""
            if has_audio:
                logging.info("Starting audio transcription")
                transcript = audio_to_text(audio_path)
                transcript_path = video_data_dir / "transcript.txt"
                save_transcript(transcript, transcript_path)
            else:
                logging.info("Skipping audio transcription for silent video")

            logging.info("Processing documents for new video")
            new_documents = load_documents(mixed_data_dir, video_id)
            documents.extend(new_documents)

            logging.info("Archiving multimedia data")
            archive_video_data(video_id, mixed_data_dir, archive_dir, lancedb_dir)
            video_ids.append(video_id)
        
        # Load documents for all existing videos
        logging.info(f"Loading documents for all videos: {video_ids}")
        for vid in video_ids:
            if vid != video_id:  # Already loaded for new video
                restore_video_data(vid, mixed_data_dir, archive_dir)
                vid_documents = load_documents(mixed_data_dir, vid)
                documents.extend(vid_documents)
        
        if documents:
            logging.info(f"Building index with documents from {len(video_ids)} videos")
            index = build_multimodal_index(documents, storage_context, video_id, image_vector_store, metadata)
        elif index is None:
            raise ValueError("No documents found and no existing index available")
        
        logging.info("Verifying persistence after processing")
        verify_persistence(lancedb_dir)

        logging.info("Setting up retriever and LLM")
        retriever = setup_retriever(index)
        llm = GoogleGenAIMultiModal(max_new_tokens=1000, temperature=0.7)

        while True:
            query = input("Enter your query (or type 'exit' to finish): ").strip()
            if query.lower() == 'exit':
                logging.info("User chose to exit")
                break
            if query:
                logging.info(f"Processing query: {query}")
                response, image_nodes = generate_response(query, retriever, llm, has_audio, metadata, video_id)
                logging.info("Saving retrieved images")
                plot_images(image_nodes, output_dir)
                print("\n" + "="*80)
                print("FINAL RESPONSE")
                print("="*80)
                print(response)
                print("="*80)
            else:
                logging.info("Empty query provided, prompting again")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        video_data_dir = mixed_data_dir / video_id
        if video_data_dir.exists():
            logging.info(f"Preserving {video_data_dir} for debugging")
        raise

if __name__ == "__main__":
    main()