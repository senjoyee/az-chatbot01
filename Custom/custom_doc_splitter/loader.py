from typing import List, Optional, Union, Dict
from pathlib import Path
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
import pymupdf4llm

class PyMuPDF4LLMMarkdownLoader(BaseLoader):
    """Custom document loader for extracting markdown-formatted text from PDF files."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        page_chunks: bool = True,
        extract_images: bool = False,
        image_path: Optional[str] = None,
        image_format: str = "png",
        dpi: int = 300,
        extract_words: bool = False
    ):
        self.file_path = Path(file_path)
        self.pages = pages
        self.page_chunks = page_chunks
        self.extract_images = extract_images
        self.image_path = image_path
        self.image_format = image_format
        self.dpi = dpi
        self.extract_words = extract_words
        
        if self.extract_images and not self.image_path:
            raise ValueError("image_path must be provided when extract_images=True")

    def load(self) -> List[Document]:
        """Load and process the PDF file into markdown format."""
        content = pymupdf4llm.to_markdown(
            doc=str(self.file_path),
            pages=self.pages,
            page_chunks=self.page_chunks,
            write_images=self.extract_images,
            image_path=self.image_path,
            image_format=self.image_format,
            dpi=self.dpi,
            extract_words=self.extract_words
        )
        
        documents = []
        
        if self.page_chunks:
            for page_num, page_content in enumerate(content):
                metadata = {
                    "source": str(self.file_path),
                    "page": page_num,
                    "file_metadata": page_content.get("metadata", {}),
                    "images": [
                        {
                            "path": f"{self.image_path}/image_{i}.{self.image_format}" if self.extract_images else None,
                            "bbox": img.get("bbox"),
                            "width": img.get("width"),
                            "height": img.get("height")
                        } for i, img in enumerate(page_content.get("images", []))
                    ],
                    "tables": page_content.get("tables", []),
                    "graphics": page_content.get("graphics", [])
                }
                
                if self.extract_words:
                    metadata["words"] = page_content.get("words", [])
                
                doc = Document(
                    page_content=page_content["text"],
                    metadata=metadata
                )
                documents.append(doc)
        else:
            metadata = {
                "source": str(self.file_path),
                "page": None,
                "file_metadata": {}
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            
        return documents