"""Document type classifier using Claude 4.5 Sonnet vision."""

import base64
from pathlib import Path
from PIL import Image
import io
import anthropic
from typing import Optional

from ..config import get_settings


class DocumentClassifier:
    """Classifies documents into types using vision-language model.
    
    This classifier is flexible and can identify any document type,
    not just a predefined set. It analyzes visual characteristics
    to provide descriptive classification.
    """
    
    CLASSIFICATION_PROMPT = """Analyze this document page and provide a classification.

Look at the visual structure and content to determine what type of technical document this is.

Common types include (but are NOT limited to):
- Operating manual / instruction guide
- Electrical wiring diagram / schematic
- P&ID (piping and instrumentation diagram)
- Pneumatic/hydraulic layout
- Mechanical assembly drawing
- Bill of materials / parts list
- Maintenance schedule
- Troubleshooting guide
- Safety documentation
- Control system diagram
- Process flow diagram

Respond with a JSON object:
{
  "document_type": "short_snake_case_name",
  "description": "Brief description of what this document contains",
  "characteristics": ["key visual feature 1", "key visual feature 2"]
}

Choose a document_type that accurately describes the content. Use snake_case.
Examples: "electrical_schematic", "operating_manual", "parts_list", "control_diagram", "safety_guide"
"""

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
    
    def classify_page(self, image: Image.Image) -> dict:
        """
        Classify a single page image.
        
        Args:
            image: PIL Image of the document page.
            
        Returns:
            Dict with document_type, description, and characteristics.
        """
        image_b64 = self._image_to_base64(image)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
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
            
            result = response.content[0].text.strip()
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```" in result:
                    result = result.split("```")[1]
                    if result.startswith("json"):
                        result = result[4:]
                
                classification = json.loads(result)
                return {
                    "document_type": classification.get("document_type", "unknown"),
                    "description": classification.get("description", ""),
                    "characteristics": classification.get("characteristics", []),
                }
            except json.JSONDecodeError:
                # Fallback: extract type from plain text
                return {
                    "document_type": "unknown",
                    "description": result[:200],
                    "characteristics": [],
                }
                
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "document_type": "unknown",
                "description": f"Classification failed: {str(e)}",
                "characteristics": [],
            }
    
    def classify_document(
        self, 
        page_images: list[Image.Image],
        sample_pages: int = 3,
    ) -> dict:
        """
        Classify an entire document by sampling multiple pages.
        
        Args:
            page_images: List of page images.
            sample_pages: Number of pages to sample for classification.
            
        Returns:
            Classification dict with most representative type.
        """
        if not page_images:
            return {"document_type": "unknown", "description": "", "characteristics": []}
        
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
            classification = self.classify_page(page_images[idx])
            if classification["document_type"] != "unknown":
                classifications.append(classification)
        
        # Return first valid classification (or combine info from multiple)
        if classifications:
            # Use the first classification but note if there are mixed types
            result = classifications[0]
            if len(set(c["document_type"] for c in classifications)) > 1:
                result["description"] += " (Note: Document contains multiple section types)"
            return result
        
        return {"document_type": "unknown", "description": "", "characteristics": []}
