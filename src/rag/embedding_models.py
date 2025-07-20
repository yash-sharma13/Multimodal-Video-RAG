import logging
import numpy as np
import torch
from PIL import Image
from typing import Any, Sequence, Callable, Optional
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom image embedding class for CLIP
class CustomClipImageEmbedding(MultiModalEmbedding):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        logging.info(f"Initializing CLIP model: {model_name}")
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()
        self._text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("CLIP model initialized successfully")

    def _get_image_embedding(self, image: Any) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        inputs = self._processor(images=[image], return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self._model = self._model.cuda()
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        embedding = embeddings[0].cpu().numpy()
        logging.debug(f"Image embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding)}")
        return embedding

    async def _aget_image_embedding(self, image: Any) -> np.ndarray:
        return self._get_image_embedding(image)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        embedding = embeddings[0].cpu().numpy()
        logging.debug(f"Text embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding)}")
        return embedding

    async def _aget_text_embedding(self, text: str) -> np.ndarray:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        logging.info(f"Generating query embedding for: {query[:50]}...")
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> np.ndarray:
        return self._get_query_embedding(query)

    def get_text_similarity(self, query: str, text: str) -> float:
        embeddings = self._text_embed_model.encode([query, text], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        logging.debug(f"Text similarity between query '{query[:20]}...' and text '{text[:20]}...': {similarity}")
        return similarity

    def get_agg_embedding_from_queries(self, queries: Sequence[str], agg_fn: Optional[Callable] = None) -> np.ndarray:
        embeddings = [self._get_query_embedding(query) for query in queries]
        if agg_fn is None:
            agg_fn = np.mean
        return agg_fn(embeddings, axis=0)

    async def aget_agg_embedding_from_queries(self, queries: Sequence[str], agg_fn: Optional[Callable] = None) -> np.ndarray:
        return self.get_agg_embedding_from_queries(queries, agg_fn) 