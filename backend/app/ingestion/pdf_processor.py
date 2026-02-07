"""PDF processing utilities for Project Mandolin."""

import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io
from typing import Generator
from dataclasses import dataclass


@dataclass
class PageImage:
    """Represents a single page extracted from a PDF."""
    page_number: int
    image: Image.Image
    width: int
    height: int
    text_content: str


class PDFProcessor:
    """Handles PDF to image conversion and text extraction."""
    
    def __init__(self, dpi: int = 150):
        """
        Initialize the PDF processor.
        
        Args:
            dpi: Resolution for rendering pages as images.
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # 72 is the default PDF resolution
    
    def get_page_count(self, pdf_path: Path) -> int:
        """Get the total number of pages in a PDF."""
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    
    def extract_pages(self, pdf_path: Path) -> Generator[PageImage, None, None]:
        """
        Extract all pages from a PDF as images with text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Yields:
            PageImage objects for each page.
        """
        doc = fitz.open(pdf_path)
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page to image
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Extract text content
                text_content = page.get_text()
                
                yield PageImage(
                    page_number=page_num + 1,  # 1-indexed
                    image=image,
                    width=pix.width,
                    height=pix.height,
                    text_content=text_content,
                )
        finally:
            doc.close()
    
    def save_page_images(
        self, 
        pdf_path: Path, 
        output_dir: Path,
        document_id: str,
    ) -> list[Path]:
        """
        Extract and save all pages as PNG images.
        
        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to save images.
            document_id: Unique document identifier.
            
        Returns:
            List of paths to saved images.
        """
        # Create document-specific directory
        doc_dir = output_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for page in self.extract_pages(pdf_path):
            image_path = doc_dir / f"page_{page.page_number}.png"
            page.image.save(image_path, "PNG")
            saved_paths.append(image_path)
            
            # Also save text content
            text_path = doc_dir / f"page_{page.page_number}.txt"
            with open(text_path, "w") as f:
                f.write(page.text_content)
        
        return saved_paths
