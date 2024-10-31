from typing import Dict, List, Optional, Any
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import hashlib
import re

class GPTSplitter(TextSplitter):
    """A document splitter that uses Azure OpenAI GPT to create logical chunks."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.0,
        cache: Optional[Dict[str, List[str]]] = None,
        api_version: str = "2023-03-15-preview",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = AzureChatOpenAI(
            model=model_name,
            api_version=api_version,
            temperature=temperature
        )
        self._cache = cache or {}
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert document splitter. Split the following text into logical chunks, preserving complete sections and context.
            Each chunk should be a complete logical unit.
            Use consistent splitting criteria like headers or major topic changes.
            Mark chunk boundaries with <<<>>> markers.
            Maintain original formatting.

            Text to split:
            {text}
            """
        )
        self.output_parser = StrOutputParser()
        self.chain = (
            {"text": RunnablePassthrough()}
            | self.prompt
            | self.model
            | self.output_parser
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text to ensure consistent splitting"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', '\n', text)
        return text

    def _get_cache_key(self, text: str) -> str:
        """Generate a deterministic cache key for the text"""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def split_text(self, text: str) -> List[Document]:
        """Split text into chunks using GPT with caching and fallback."""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            return [Document(page_content=chunk) for chunk in self._cache[cache_key]]
        
        normalized_text = self._normalize_text(text)
        
        try:
            chunks = self.chain.invoke(normalized_text).split("<<<")[1:]
            chunks = [chunk.strip(">>>").strip() for chunk in chunks]
            self._cache[cache_key] = chunks
            return [Document(page_content=chunk) for chunk in chunks]
            
        except Exception as e:
            print(f"GPT splitting failed: {e}")
            chunks = self._fallback_split(normalized_text)
            self._cache[cache_key] = chunks
            return [Document(page_content=chunk) for chunk in chunks]
    
    def _fallback_split(self, text: str) -> List[str]:
        """Fallback splitting strategy using headers"""
        header_pattern = r'\n\d+\.\s+[A-Z]|\n[A-Z][^a-z\n]+\n'
        chunks = re.split(header_pattern, text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks