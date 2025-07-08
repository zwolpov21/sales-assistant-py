from pydantic import BaseModel
from typing import Optional


class ScrapeRequest(BaseModel):
    """
    Request model for web scraping.
    """
    url: str
    output_formats: Optional[list[str]] = ["html", "markdown"]  # Optional list of output formats, e.g., ["html", "markdown"]

class GeminiSearchRequest(BaseModel):
    """
    Request model for performing a web search using Google Gemini.
    """
    query: str
    grounding: Optional[bool] = True  # Whether to use grounding tools for real-time information
    model:  Optional[str]  # Model to use for content generation
    in_text_citations: Optional[bool] = False  # Whether to include in-text citations
