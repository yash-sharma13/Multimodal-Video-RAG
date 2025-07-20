import logging
import os
import asyncio
from typing import List, Optional, Any, AsyncGenerator
from pydantic import ConfigDict
import google.generativeai as genai
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.llms import LLMMetadata
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, ChatResponse, ChatResponseGen, ChatMessage
from dotenv import load_dotenv

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Custom GoogleGenAIMultiModal class
class GoogleGenAIMultiModal(MultiModalLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, model_name: str = "gemini-1.5-flash-002", max_new_tokens: int = 1000, temperature: float = 0.7):
        super().__init__()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY environment variable is not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name, generation_config={"temperature": temperature})
            logging.info(f"Initialized GoogleGenAIMultiModal with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize GenerativeModel: {str(e)}")
            raise
        self._metadata = LLMMetadata(
            context_window=1000000,
            num_output=max_new_tokens,
            model_name=model_name,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return self._metadata

    def complete(self, prompt: str, image_documents: Optional[List] = None, **kwargs: Any) -> CompletionResponse:
        parts = [{"text": prompt}]
        if image_documents:
            for doc in image_documents:
                image_path = getattr(doc, 'image_path', doc.metadata.get('image_path') if hasattr(doc, 'metadata') else None)
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        image_data = img_file.read()
                    parts.append({"inline_data": {"mime_type": "image/png", "data": image_data}})
                elif hasattr(doc, 'image'):
                    parts.append({"inline_data": {"mime_type": "image/png", "data": doc.image}})
                else:
                    logging.warning(f"Invalid image document: {doc}")
        try:
            response = self._model.generate_content(parts, **kwargs)
            return CompletionResponse(text=response.text, raw=response)
        except Exception as e:
            logging.error(f"Error in GoogleGenAIMultiModal.complete: {str(e)}")
            raise

    async def acomplete(self, prompt: str, image_documents: Optional[List] = None, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, image_documents, **kwargs))

    def stream_complete(self, prompt: str, image_documents: Optional[List] = None, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, image_documents, **kwargs)
        yield CompletionResponse(text=response.text, raw=response)

    async def astream_complete(self, prompt: str, image_documents: Optional[List] = None, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        response = await self.acomplete(prompt, image_documents, **kwargs)
        yield CompletionResponse(text=response.text, raw=response)

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        response = self.complete(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role="assistant", content=response.text), raw=response)

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        response = self.chat(messages, **kwargs)
        yield ChatResponse(message=ChatMessage(role="assistant", content=response.message.content), raw=response)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        response = await self.achat(messages, **kwargs)
        yield ChatResponse(message=ChatMessage(role="assistant", content=response.message.content), raw=response) 