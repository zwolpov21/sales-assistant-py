from fastapi import FastAPI, Request
from pydantic import BaseModel
from pinecone import Pinecone, EmbedModel
from openai import OpenAI
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain.schema import Document
import numpy as np
import os
from collections import OrderedDict
from dotenv import load_dotenv
from typing import Optional
from app.services.pinecone_service import PineconeService
from app.services.openai_service import OpenAIService
from app.routers import query, web_search
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Optional: define an app-specific logger
logger = logging.getLogger("query_pipeline.main")
logger.info("Logging configured in main.py")

# Load development env vars if necessary
if os.getenv("RENDER") != "true":
    load_dotenv()

# FastAPI application instance
app = FastAPI()
app.include_router(router=query.router)
app.include_router(router=web_search.router)




@app.get("/")
async def root():
    return {"status": "ok"}


