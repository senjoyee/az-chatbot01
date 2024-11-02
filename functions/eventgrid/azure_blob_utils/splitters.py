import logging
import hashlib
import re
from typing import List, Dict, Optional, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters.base import TextSplitter
from langchain_openai import AzureChatOpenAI


class GPTSplitter(TextSplitter):
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.0,
        cache: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = AzureChatOpenAI(
            model=model_name,
            api_version="2023-03-15-preview",
            temperature=temperature
        )
        self._cache = cache or {}
        
        self.split_prompt = ChatPromptTemplate.from_template(
            """You are an expert document splitter. Split the following text into logical chunks, preserving complete sections and context.
            Each chunk should be a complete logical unit.
            Use consistent splitting criteria like headers or major topic changes.
            Mark chunk boundaries with <<<>>> markers.
            Maintain original formatting.

            Text to split:
            {text}
            """
        )

        self.customer_prompt = ChatPromptTemplate.from_template(
            """Extract the customer company name from the text. If multiple variations exist, use the formal name.
            Common patterns:
            - "Customer: [Name]"
            - "[Name] has engaged"
            - "On behalf of [Name]"
            - References to "the Customer [Name]"

            Return ONLY the company name, nothing else.
            If no customer is found, return "Unknown".

            Text:
            {text}
            """
        )
        
        self.output_parser = StrOutputParser()
        
        self.split_chain = (
            {"text": RunnablePassthrough()}
            | self.split_prompt
            | self.model
            | self.output_parser
        )
        
        self.customer_chain = (
            {"text": RunnablePassthrough()}
            | self.customer_prompt
            | self.model
            | self.output_parser
        )

    def _detect_customer(self, text: str) -> str:
        try:
            customer_name = self.customer_chain.invoke(text[:2000]).strip()
            return customer_name if customer_name else "Unknown"
        except Exception as e:
            logging.error(f"Customer detection failed: {e}")
            return "Unknown"

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', '\n', text)
        return text

    def _get_cache_key(self, text: str) -> str:
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def split_text(self, text: str, source: str = None) -> List[Document]:
        if not source or source.isspace():
            source = "unknown"
            logging.warning("Empty source provided to GPTSplitter, using 'unknown'")

        cache_key = self._get_cache_key(text)
        customer_name = self._detect_customer(text)
        
        if cache_key in self._cache:
            return [
                Document(
                    page_content=chunk, 
                    metadata={
                        "customer_name": customer_name,
                        "source": source
                    }
                ) for chunk in self._cache[cache_key]
            ]
        
        normalized_text = self._normalize_text(text)
        
        try:
            chunks = self.split_chain.invoke(normalized_text).split("<<<")[1:]
            chunks = [chunk.strip(">>>").strip() for chunk in chunks]
            self._cache[cache_key] = chunks
            
            return [
                Document(
                    page_content=chunk, 
                    metadata={
                        "customer_name": customer_name,
                        "source": source
                    }
                ) for chunk in chunks
            ]
            
        except Exception as e:
            logging.error(f"GPT splitting failed: {e}")
            chunks = self._fallback_split(normalized_text)
            self._cache[cache_key] = chunks
            return [
                Document(
                    page_content=chunk, 
                    metadata={
                        "customer_name": customer_name,
                        "source": source
                    }
                ) for chunk in chunks
            ]
    
    def _fallback_split(self, text: str) -> List[str]:
        header_pattern = r'\n\d+\.\s+[A-Z]|\n[A-Z][^a-z\n]+\n'
        chunks = re.split(header_pattern, text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks