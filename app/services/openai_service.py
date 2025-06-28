from openai import OpenAI
import numpy as np
from app.schemas import openai_schemas
from pydantic import BaseModel



class OpenAIService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def get_openai_embeddings(self, query: str, model: str = "text-embedding-3-small") -> list[float]:
        '''
        Gets dense embedding of inputted query.
        Wraps OpenAI's 'text-embedding-3-small' model.
        Returns: {"values": list of vectors}
        '''
        response = self.client.embeddings.create(
            input=query,
            model=model
        )
        return {"values": response.data[0].embedding}
    
    def get_chat_completion(
        self, 
        messages: list[dict], 
        model: str = "gpt-4.1-mini",
        response_model: type[BaseModel] = openai_schemas.StandardChatResponse
    ) -> str:
        """
        Gets chat completion from OpenAI's chat models.
        Wraps OpenAI's 'gpt-3.5-turbo' model.
        Returns: string (the assistant's reply)
        """
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": messages.get("system")
                },
                {
                    "role": "user",
                    "content": messages.get("user")
                }
            ],
            text_format=response_model
        )
        return {"output": response.output_parsed}
