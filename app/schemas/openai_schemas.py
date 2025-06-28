from pydantic import BaseModel
from typing import Optional


class GenerateQueriesReturnFormat(BaseModel):
    queries: list[str]

class StandardChatResponse(BaseModel):
    output: str