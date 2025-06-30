from pydantic import BaseModel
from typing import Optional


class ScrapeRequest(BaseModel):
    """
    Request model for web scraping.
    """
    url: str
    output_formats: Optional[list[str]] = ["html", "markdown"]  # Optional list of output formats, e.g., ["html", "markdown"]
