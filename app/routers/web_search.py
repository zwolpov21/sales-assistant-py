from fastapi import APIRouter, HTTPException
import logging
from app.services.firecrawl_service import FirecrawlService
from app.services.gemini_service import GeminiService
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

# GeminiService
gemini_client = GeminiService(api_key=os.getenv("GEMINI_API_KEY"))

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


@router.post("/web_search/gemini_search")
async def gemini_search(request_body: web_search_schemas.GeminiSearchRequest):
    """
    Perform a web search using Google Gemini.
    
    Returns the search results in a structured format.
    """
    try:
        logger.info(f"Performing Gemini search for query: {request_body.query}...")
        gemini_client.config_web_search_tool()
        response = gemini_client.get_completion(web_search=True, query=request_body.query, model=request_body.model)
        logger.info("Gemini search completed successfully.")

        # Add both in text and list citations to the response if indicated
        if request_body.in_text_citations:
            if not response.candidates or not response.candidates[0].grounding_metadata:
                logger.warning("No grounding metadata found in the response. In-text citations will not be added.")
                return {"search_response": response.text}
            logger.info("Adding text and list citations to the response...")
            response_cited = gemini_client.full_citations(response)
            logger.info("Citations added successfully.")
            return {"search_response": response_cited.get("text", ""), "citations": response_cited.get("citations", [])}

        # If no in-text citations, add only list citations
        else:
            logger.info("No in-text citations requested. Returning response with just list citations.")
            if not response.candidates or not response.candidates[0].grounding_metadata:
                logger.warning("No grounding metadata found in the response. Returning response without citations.")
                return {"search_response": response.text}
            citations: list[str] = gemini_client.list_citations(response)
            logger.info("List citations added successfully.")

            return {"search_response": response.text, "citations": citations}

    except Exception as e:
        logger.error(f"Error performing Gemini search: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred during the Gemini search: {str(e)}")