from fastapi import APIRouter, HTTPException
import logging
from app.services.firecrawl_service import FirecrawlService
from app.schemas import web_search_schemas
import os
from dotenv import load_dotenv


# router
router = APIRouter()

# Load development env vars if necessary
# Absolutely no idea why, but this line is necessary here
# even though it is also in main.py
if os.getenv("RENDER") != "true":
    load_dotenv()

# FirecrawlService 
firecrawl_client = FirecrawlService(api_key=os.getenv("FIRECRAWL_KEY"))

# logging
logger = logging.getLogger(__name__)

@router.post("/web_search/scrape_url")
async def scrape_url(request_body: web_search_schemas.ScrapeRequest):
    """
    Scrape the content from a given URL using Firecrawl.
    
    Outputs the scraped content in a specified format.
    """
    try:
        logger.info(f"Scraping URL: {request_body.url} with formats: {request_body.output_formats}...")
        scraped_content = firecrawl_client.scrape_url(request_body.url, request_body.output_formats)
        logger.info("Content scraped successfully.")

        # Handle output formats (there may be multiple!)
        full_content: dict = {}
        for format in request_body.output_formats:
            if format == "html":
                full_content["html"] = scraped_content.html
            elif format == "markdown":
                full_content["markdown"] = scraped_content.markdown
            else:
                logger.warning(f"Unsupported format requested: {format}")

        return {"scraped_content": full_content}
    except Exception as e:
        logger.error(f"Error scraping URL {request_body.url}: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred when scraping URL: {str(e)}")
