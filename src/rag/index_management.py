import logging
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, ImageNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.embedding_models import CustomClipImageEmbedding
from utils.image_ocr import extract_ocr_text
import lancedb
import re

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up settings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.image_embed_model = CustomClipImageEmbedding(model_name="openai/clip-vit-large-patch14")
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

def load_documents(data_dir: Path, video_id: str) -> List:
    try:
        video_data_dir = data_dir / video_id
        if not video_data_dir.exists():
            raise ValueError(f"Video data directory {video_data_dir} does not exist")
        for file in video_data_dir.glob("*"):
            if file.name not in ['transcript.txt'] and not file.name.endswith('.png'):
                file.unlink()
                logging.debug(f"Removed non-relevant file: {file}")
        transcript_path = video_data_dir / "transcript.txt"
        documents = []
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            if transcript:
                doc = SimpleDirectoryReader(input_files=[transcript_path]).load_data()[0]
                doc.metadata['video_id'] = video_id
                doc.metadata['ocr_text'] = ""
                doc.metadata['ocr_quality'] = 0.0
                documents.append(doc)
                logging.info(f"Loaded transcript from {transcript_path}, length={len(transcript)}")
        image_docs = SimpleDirectoryReader(input_dir=str(video_data_dir), required_exts=['.png']).load_data()
        for doc in image_docs:
            if hasattr(doc, 'image_path') and doc.image_path:
                ocr_text, ocr_quality = extract_ocr_text(doc.image_path)
                doc.metadata['image_path'] = str(Path(doc.image_path).resolve())
                doc.metadata['ocr_text'] = ocr_text
                doc.metadata['ocr_quality'] = ocr_quality
                doc.metadata['video_id'] = video_id
                documents.append(doc)
                logging.debug(f"Loaded image document {doc.image_path}, ocr_text={ocr_text[:50]}...")
        if not documents:
            logging.warning(f"No documents loaded for video {video_id}")
        else:
            logging.info(f"Loaded {len(documents)} documents from {video_data_dir}")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents: {str(e)}")
        raise

def build_multimodal_index(
    documents: List,
    storage_context: StorageContext,
    video_id: str,
    image_vector_store: LanceDBVectorStore,
    metadata: Optional[dict] = None # Added metadata parameter
) -> MultiModalVectorStoreIndex:
    try:
        logging.info(f"Building index with {len(documents)} documents for video {video_id}")
        text_splitter = Settings.text_splitter
        text_nodes = []
        image_nodes = []
        for doc in documents:
            if hasattr(doc, 'text') and doc.text:
                cleaned_text = re.sub(r'\s+', ' ', doc.text.strip())
                chunks = text_splitter.split_text(cleaned_text)
                for i, chunk in enumerate(chunks):
                    node_id = f"{video_id}_text_{doc.id_}_{i}"
                    node = TextNode(
                        text=chunk,
                        id_=node_id,
                        metadata={
                            "doc_id": doc.id_,
                            "video_id": video_id,
                            "ocr_text": doc.metadata.get('ocr_text', ''),
                            "ocr_quality": doc.metadata.get('ocr_quality', 0.0)
                        }
                    )
                    node.embedding = Settings.embed_model.get_text_embedding(chunk)
                    logging.debug(f"TextNode {node_id}: text={chunk[:50]}..., embedding_shape={np.array(node.embedding).shape}")
                    text_nodes.append(node)
            if hasattr(doc, 'image_path') and doc.image_path:
                node_id = f"{video_id}_image_{doc.id_}"
                image_path = str(Path(doc.image_path).resolve())
                if not os.path.exists(image_path):
                    logging.warning(f"Image path {image_path} does not exist")
                    continue
                ocr_text = doc.metadata.get('ocr_text', '')
                ocr_quality = doc.metadata.get('ocr_quality', 0.0)
                node = ImageNode(
                    image_path=image_path,
                    id_=node_id,
                    metadata={
                        "doc_id": doc.id_,
                        "video_id": video_id,
                        "ocr_text": ocr_text,
                        "ocr_quality": ocr_quality,
                        "image_path": image_path
                    }
                )
                node.embedding = Settings.image_embed_model._get_image_embedding(image_path)
                logging.debug(f"ImageNode {node_id}: image_path={image_path}, ocr_text={ocr_text[:50]}..., ocr_quality={ocr_quality}")
                image_nodes.append(node)

        logging.info(f"Created {len(text_nodes)} text nodes and {len(image_nodes)} image nodes for video {video_id}")

        # Validate embeddings
        for node in text_nodes + image_nodes:
            if not hasattr(node, 'embedding') or node.embedding is None:
                logging.error(f"Node {node.id_} is missing embedding")
                raise ValueError(f"Node {node.id_} is missing embedding")

        # Add nodes to respective vector stores
        if text_nodes:
            storage_context.vector_store.add(text_nodes)
            logging.info(f"Added {len(text_nodes)} text nodes to text_collection")
        if image_nodes:
            image_vector_store.add(image_nodes)
            logging.info(f"Added {len(image_nodes)} image nodes to image_collection")

        index = MultiModalVectorStoreIndex(
            nodes=text_nodes + image_nodes,
            storage_context=storage_context,
            image_vector_store=image_vector_store,
            image_embed_model=Settings.image_embed_model
        )
        logging.info(f"Index built successfully with {len(text_nodes + image_nodes)} nodes")
        return index
    except Exception as e:
        logging.error(f"Error building index: {str(e)}")
        raise

def load_existing_index(lancedb_dir: Path, storage_context: StorageContext, image_vector_store: LanceDBVectorStore) -> Optional[MultiModalVectorStoreIndex]:
    try:
        db = lancedb.connect(str(lancedb_dir))
        text_nodes = []
        image_nodes = []
        video_ids = set()
        for table_name, node_type in [("text_collection", TextNode), ("image_collection", ImageNode)]:
            if table_name in db.table_names():
                table = db.open_table(table_name)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    metadata = row['metadata']
                    video_id = metadata.get('video_id', 'unknown')
                    video_ids.add(video_id)
                    node_kwargs = {
                        "id_": row['id'],
                        "metadata": metadata,
                        "embedding": row['vector']
                    }
                    if node_type == TextNode:
                        node_kwargs["text"] = row['text']
                    elif node_type == ImageNode:
                        image_path = metadata.get('image_path', '')
                        if not image_path or not os.path.exists(image_path):
                            logging.warning(f"Image path {image_path} does not exist, skipping node")
                            continue
                        node_kwargs["image_path"] = image_path
                    node = node_type(**node_kwargs)
                    logging.debug(f"Loaded node {node.id_}: metadata={metadata}")
                    if node_type == TextNode:
                        text_nodes.append(node)
                    else:
                        image_nodes.append(node)
        if text_nodes or image_nodes:
            index = MultiModalVectorStoreIndex(
                nodes=text_nodes + image_nodes,
                storage_context=storage_context,
                image_vector_store=image_vector_store,
                image_embed_model=Settings.image_embed_model
            )
            logging.info(f"Loaded existing index with {len(text_nodes)} text nodes and {len(image_nodes)} image nodes from {lancedb_dir}, covering videos: {video_ids}")
            return index
        logging.info(f"No existing index found at {lancedb_dir}")
        return None
    except Exception as e:
        logging.error(f"Error loading existing index: {str(e)}")
        raise 