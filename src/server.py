from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from utils.common_utils import create_folders, validate_youtube_url, get_video_id
from db.lancedb_utils import setup_vector_stores, check_video_exists
from rag.index_management import load_existing_index, load_documents, build_multimodal_index
from utils.video_processing import download_video, video_to_images, video_to_audio, audio_to_text, save_transcript
from rag.llm_models import GoogleGenAIMultiModal
from rag.retriever_response import setup_retriever, generate_response, plot_images
from data_management.archive_utils import archive_video_data
from constants import OUTPUT_DIR, MIXED_DATA_DIR, ARCHIVE_DIR, LANCEDB_DIR

from llama_index.core import StorageContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize global variables - moved to constants.py or removed if no longer needed
output_dir = OUTPUT_DIR
create_folders(output_dir) # Ensure folders are created on app startup
mixed_data_dir = MIXED_DATA_DIR
archive_dir = ARCHIVE_DIR
lancedb_dir = LANCEDB_DIR

# Create FastAPI app
app = FastAPI(title="Video RAG API", description="API for video-based RAG system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory - serve directly from output directory
app.mount("/static", StaticFiles(directory=str(output_dir)), name="static")

# Global state
current_video_id = None
current_retriever = None
current_llm = None
current_metadata = None
has_audio = True

class VideoURL(BaseModel):
    url: str

class Query(BaseModel):
    query: str

class Response(BaseModel):
    response: str
    video_id: str
    image_paths: List[str]

@app.post("/process-video")
async def process_video(video: VideoURL):
    global current_video_id, current_retriever, current_llm, current_metadata, has_audio
    
    try:
        if not validate_youtube_url(video.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        video_id = get_video_id(video.url)
        logging.info(f"Processing video with ID: {video_id}")

        from pytubefix import YouTube
        yt = YouTube(video.url)
        current_metadata = {
            "title": yt.title or "Unknown",
            "author": yt.author or "Unknown",
            "length": yt.length or 0,
            "publish_date": str(yt.publish_date) if yt.publish_date else "Unknown"
        }
        
        # Set up vector stores
        text_store, image_store = setup_vector_stores(lancedb_dir)
        storage_context = StorageContext.from_defaults(vector_store=text_store)
        
        # Check if video exists
        if check_video_exists(video_id, lancedb_dir):
            logging.info(f"Video {video_id} already processed, loading existing index")
            index = load_existing_index(lancedb_dir, storage_context, image_store)
            if not index:
                raise HTTPException(status_code=500, detail="Failed to load existing index")
            return {"message": "Video already exists", "video_id": video_id}
        else:
            # Download and process video
            video_path = download_video(video.url, output_dir)
            video_data_dir = mixed_data_dir / video_id
            video_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            video_to_images(video_path, video_data_dir)
            
            # Extract audio
            audio_path = video_data_dir / "audio.wav"
            has_audio = video_to_audio(video_path, audio_path)
            
            # Transcribe audio if available
            if has_audio:
                transcript = audio_to_text(audio_path)
                transcript_path = video_data_dir / "transcript.txt"
                save_transcript(transcript, transcript_path)
            
            # Load and process documents
            documents = load_documents(mixed_data_dir, video_id)
            if not documents:
                raise HTTPException(status_code=500, detail="No documents loaded")
            
            # Build index
            index = build_multimodal_index(documents, storage_context, video_id, image_store, current_metadata)
        
        # Set up retriever and LLM
        current_retriever = setup_retriever(index)
        current_llm = GoogleGenAIMultiModal(max_new_tokens=1000, temperature=0.7)
        current_video_id = video_id
        
        return {"message": "Video processed successfully", "video_id": video_id}
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-query", response_model=Response)
async def chat_query(query: Query):
    global current_video_id, current_retriever, current_llm, current_metadata, has_audio
    
    try:
        logging.info(f"Received query: {query.query}")
        
        if not current_retriever or not current_llm:
            logging.error("No video has been processed yet")
            raise HTTPException(
                status_code=400, 
                detail="Please process a video first using the /process-video endpoint"
            )
        
        if not current_video_id:
            logging.error("No current video ID found")
            raise HTTPException(
                status_code=400,
                detail="No video is currently being processed"
            )
        
        # Generate response
        response, image_nodes = generate_response(
            query.query,
            current_retriever,
            current_llm,
            has_audio,
            current_metadata
        )
        
        # Save retrieved images
        image_paths = []
        if image_nodes:
            plot_images(image_nodes, output_dir)
            image_paths = [str(output_dir / f"retrieved_image_{i}.png") 
                         for i in range(1, min(len(image_nodes) + 1, 6))]
        
        logging.info(f"Generated response with {len(image_paths)} images")
        
        return Response(
            response=response,
            video_id=current_video_id,
            image_paths=image_paths
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/archive-video")
async def archive_video(video_id: str):
    try:
        archive_video_data(video_id, mixed_data_dir, archive_dir, lancedb_dir)
        return {"message": f"Video {video_id} archived successfully"}
    except Exception as e:
        logging.error(f"Error archiving video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)