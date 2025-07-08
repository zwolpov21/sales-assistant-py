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
