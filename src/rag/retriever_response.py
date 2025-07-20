import logging
import os
import json
import time
from typing import Any, List, Optional
import numpy as np
from PIL import Image
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer, util
from rag.llm_models import GoogleGenAIMultiModal
from data_management.archive_utils import restore_video_data
from pathlib import Path
import lancedb

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# DistilBERT semantic scoring
def compute_semantic_score(query: str, text: str) -> float:
    if not text.strip():
        return 0.0
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([query, text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def setup_retriever(index: MultiModalVectorStoreIndex):
    try:
        retriever = index.as_retriever(similarity_top_k=20, image_similarity_top_k=5)
        logging.info("Retriever set up with similarity_top_k=20, image_similarity_top_k=5")
        return retriever
    except Exception as e:
        logging.error(f"Error setting up retriever: {str(e)}")
        raise

def generate_response(
    query: str,
    retriever: Any,
    llm: GoogleGenAIMultiModal,
    has_audio: bool,
    metadata: dict,
    video_id: Optional[str] = None
) -> tuple:
    try:
        start_time = time.time()
        # Restore data for all videos in lancedb
        db = lancedb.connect("output/lancedb")
        video_ids = []
        if 'image_collection' in db.table_names():
            table = db.open_table("image_collection")
            df = table.to_pandas()
            video_ids = list(set(row['metadata']['video_id'] for _, row in df.iterrows() if 'video_id' in row['metadata']))
        logging.info(f"Restoring data for videos: {video_ids}")
        for vid in video_ids:
            restore_video_data(vid, Path("output/mixed_data"), Path("output/video_archives"))
            video_data_dir = Path("output/mixed_data") / vid
            if video_data_dir.exists():
                logging.debug(f"Verified data exists for video {vid} at {video_data_dir}")
            else:
                logging.warning(f"Data for video {vid} not restored properly")
        
        retrieval_results = retriever.retrieve(query)
        logging.debug(f"Retrieved {len(retrieval_results)} nodes")

        text_nodes = [node for node in retrieval_results if hasattr(node, 'node') and isinstance(node.node, TextNode)]
        image_nodes = [node for node in retrieval_results if hasattr(node, 'node') and isinstance(node.node, ImageNode)]

        video_ids = set(node.node.metadata.get('video_id') for node in retrieval_results if hasattr(node, 'node'))
        logging.info(f"Retrieved nodes from videos: {video_ids}")

        scores = []
        for node in image_nodes:
            image_path = node.node.metadata.get('image_path', '')
            if not image_path or not os.path.exists(image_path):
                logging.warning(f"Invalid image path: {image_path}")
                continue

            ocr_text = node.node.metadata.get('ocr_text', '')
            ocr_quality = node.node.metadata.get('ocr_quality', 0.0)
            clip_score = node.score

            semantic_score = compute_semantic_score(query, ocr_text)
            ocr_score = Settings.image_embed_model.get_text_similarity(query, ocr_text) if ocr_quality > 0.5 else 0.0

            combined_score = (
                0.5 * clip_score + 0.4 * semantic_score + 0.1 * ocr_score
                if ocr_quality > 0.5 else
                0.6 * clip_score + 0.4 * semantic_score
            )
            scores.append((node, combined_score, clip_score, semantic_score, ocr_score))
            logging.debug(f"Image node {node.node.id_}: combined_score={combined_score:.3f}, ocr_text={ocr_text[:50]}...")

        video_duration = metadata.get('length', 0) / 60
        if video_duration <= 5:
            num_images = 5
        elif video_duration <= 30:
            num_images = 8
        else:
            num_images = 10

        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[:max(1, min(num_images, len(scores)))]

        text_similarities = []
        for node in text_nodes:
            sentence_score = Settings.image_embed_model.get_text_similarity(query, node.node.text)
            semantic_score = compute_semantic_score(query, node.node.text)
            combined_text_score = 0.6 * sentence_score + 0.4 * semantic_score
            text_similarities.append(combined_text_score)
            logging.debug(f"Text node {node.node.id_}: combined_score={combined_text_score:.3f}, text={node.node.text[:50]}...")

        max_combined_score = max([s[1] for s in scores]) if scores else 0.0
        max_text_similarity = max(text_similarities) if text_similarities else 0.0

        relevant_text_nodes = len([s for s in text_similarities if s > 0.1])
        relevant_image_nodes = len([s for s in scores if s[1] > 0.1])

        logging.debug(f"Relevance: max_combined_score={max_combined_score:.3f}, max_text_similarity={max_text_similarity:.3f}, relevant_text_nodes={relevant_text_nodes}, relevant_image_nodes={relevant_image_nodes}")

        if max_combined_score < 0.1 and max_text_similarity < 0.1 and relevant_text_nodes == 0 and relevant_image_nodes < 1:
            logging.info(f"Query '{query}' deemed irrelevant")
            return f"No relevant content found for query '{query}'. Please try a different question related to the video.", [node[0] for node in top_nodes[:1]]

        image_documents = [
            ImageNode(image_path=str(node[0].node.metadata.get('image_path', '')))
            for node in top_nodes if os.path.exists(node[0].node.metadata.get('image_path', ''))
        ]

        context = []
        if has_audio and text_nodes:
            context.append("\n".join([f"[Video ID: {node.node.metadata['video_id']}] {node.node.text}" for node in text_nodes]))
        context.append("\n".join([
            f"[Video ID: {node[0].node.metadata['video_id']}] {node[0].node.metadata.get('ocr_text', '')}"
            for node in top_nodes if node[0].node.metadata.get('ocr_quality', 0.0) > 0.3
        ]))
        context_str = "\n".join(context) or f"No transcript or OCR text available. Video metadata: {json.dumps(metadata, indent=2)}"
        metadata_str = json.dumps(metadata, indent=2)

        qa_tmpl_str = (
            "You are an expert assistant tasked with providing a detailed, accurate, and comprehensive response to the query based solely on the provided video context, including transcribed text, OCR-extracted text, and relevant images. Do not use external knowledge. Your response should be 300-500 words, structured with an introduction, main points, and a conclusion. Synthesize information from text, OCR, and images, explicitly describing any relevant visuals (e.g., text, charts, diagrams, timestamps) and referencing video IDs (from metadata) to clarify sources. If multiple videos are relevant, compare and contrast their content. Ensure the response is precise, avoids speculation, and directly addresses the query. If the query cannot be answered with the provided context, clearly state so and explain why.\n"
            "---------------------\n"
            "Context: {context_str}\n"
            "---------------------\n"
            "Video metadata: {metadata_str}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        prompt = qa_tmpl_str.format(context_str=context_str, metadata_str=metadata_str, query_str=query)
        response = llm.complete(prompt=prompt, image_documents=image_documents)
        logging.info(f"Response generated in {time.time() - start_time:.2f} seconds")
        return response.text, [node[0] for node in top_nodes]

    except Exception as e:
        logging.error(f"Error in response: {str(e)}")
        raise

def plot_images(image_nodes: List, output_dir: Path):
    try:
        if not image_nodes:
            logging.warning("No suitable image nodes provided for plotting")
            return

        valid_image_nodes = []
        for node in image_nodes:
            inner_node = node.node if hasattr(node, 'node') else node
            if not isinstance(inner_node, ImageNode):
                logging.debug(f"Invalid node type: {type(inner_node).__name__}")
                continue
            image_path = inner_node.metadata.get('image_path', '')
            if image_path and os.path.exists(image_path):
                valid_image_nodes.append((inner_node, image_path))
            else:
                logging.warning(f"Invalid image path: {image_path}")

        if not valid_image_nodes:
            logging.warning("No valid image paths found")
            return

        num_images = min(len(valid_image_nodes), 5)
        for i, (img_node, image_path) in enumerate(valid_image_nodes[:num_images], 1):
            output_path = output_dir / f"retrieved_image_{i}.png"
            img = Image.open(image_path)
            img.save(output_path, 'PNG')
            logging.info(f"Saved image {i} to {output_path}")

    except Exception as e:
        logging.error(f"Error saving images: {str(e)}")
        raise 