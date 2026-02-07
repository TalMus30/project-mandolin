"""Document type classifier using Claude 4.5 Sonnet vision."""

import base64
from pathlib import Path
from PIL import Image
import io
import anthropic
from typing import Optional

from ..config import get_settings
from ..models import DocumentType


class DocumentClassifier:
    """Classifies documents into types using vision-language model."""
    
    CLASSIFICATION_PROMPT = """Analyze this document page and classify it into one of these categories:

1. MANUAL - Operating instructions, procedures, maintenance guides with mostly text
2. ELECTRICAL_SCHEMATIC - Electrical wiring diagrams, circuit diagrams with symbols and connections
3. MEDIA_LAYOUT - P&ID diagrams, pneumatic layouts, water/fluid line diagrams, physical equipment layouts

Look for these indicators:
- MANUAL: Numbered steps, paragraphs of text, tables of specifications
- ELECTRICAL_SCHEMATIC: Electrical symbols, wire connections, terminal blocks, relay logic
- MEDIA_LAYOUT: Flow lines, valves, tanks, pumps, physical component placement

Respond with ONLY one of these exact words: MANUAL, ELECTRICAL_SCHEMATIC, MEDIA_LAYOUT, or UNKNOWN"""

    def __init__(self):
        """Initialize the classifier with Anthropic client."""
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.vlm_model
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def classify_page(self, image: Image.Image) -> DocumentType:
        """
        Classify a single page image.
        
        Args:
            image: PIL Image of the document page.
            
        Returns:
            DocumentType classification.
        """
        image_b64 = self._image_to_base64(image)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self.CLASSIFICATION_PROMPT,
                            },
                        ],
                    }
                ],
            )
            
            result = response.content[0].text.strip().upper()
            
            # Map response to DocumentType
            if "MANUAL" in result:
                return DocumentType.MANUAL
            elif "ELECTRICAL" in result:
                return DocumentType.ELECTRICAL_SCHEMATIC
            elif "MEDIA" in result or "P&ID" in result or "LAYOUT" in result:
                return DocumentType.MEDIA_LAYOUT
            else:
                return DocumentType.UNKNOWN
                
        except Exception as e:
            print(f"Classification error: {e}")
            return DocumentType.UNKNOWN
    
    def classify_document(
        self, 
        page_images: list[Image.Image],
        sample_pages: int = 3,
    ) -> DocumentType:
        """
        Classify an entire document by sampling multiple pages.
        
        Args:
            page_images: List of page images.
            sample_pages: Number of pages to sample for classification.
            
        Returns:
            Most common DocumentType from sampled pages.
        """
        if not page_images:
            return DocumentType.UNKNOWN
        
        # Sample pages from beginning, middle, and end
        total_pages = len(page_images)
        sample_indices = [0]
        
        if total_pages > 1:
            sample_indices.append(total_pages // 2)
        if total_pages > 2:
            sample_indices.append(total_pages - 1)
        
        # Limit to sample_pages
        sample_indices = sample_indices[:sample_pages]
        
        # Classify sampled pages
        classifications = []
        for idx in sample_indices:
            doc_type = self.classify_page(page_images[idx])
            classifications.append(doc_type)
        
        # Return most common classification (excluding UNKNOWN if possible)
        non_unknown = [c for c in classifications if c != DocumentType.UNKNOWN]
        if non_unknown:
            return max(set(non_unknown), key=non_unknown.count)
        
        return DocumentType.UNKNOWN
