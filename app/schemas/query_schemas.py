from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Optional, Literal, Annotated

class Namespace(str, Enum):
    MAIN = "main"
    ALL_MEETINGS = "all_meetings"

    def __str__(self):
        return self.value

"""
Request model for Hybrid MMR retrieval from Pinecone
"""
class HybridMMRRequest(BaseModel):
    """
    Request model for Hybrid MMRR.
    """
    query: str
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    lambda_mult: float = 0.5
    mmr_top_k: int = 20

class DenseRerankRequest(BaseModel):
    query: str
    top_k: int = 100
    top_n: int = 20
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    type: Optional[str] = None  # Optional metadata filter
    rerank_model: Optional[str] = "bge-reranker-v2-m3"
    
    @model_validator(mode="after")
    def _validate(cls, values):
        """
        Validate that 'type' is valid for the given 'namespace'.
        """
        return validate_type_vs_namespace(values)

class DenseQueryRequest(BaseModel):
    query: str
    top_k: int = 20
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    type: Optional[str] = None  # Optional metadata filter

    @model_validator(mode="after")
    def _validate(cls, values):
        """
        Validate that 'type' is valid for the given 'namespace'.
        """
        return validate_type_vs_namespace(values)

"""
Supports multiple queries in a single request.
Optionally rerank results by including a top_n parameter.
Optionally filter by data 'type'.
"""
class DenseMultiQueryRequest(BaseModel):
    main_query: str
    num_queries: int = 3
    top_k: int = 100 # For each query!
    top_n: Optional[int] = 20 # For reranking each query's results, if None, reranks at end only
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    type: Optional[str] = None  # Optional metadata filter
    rerank_model: Optional[str] = "bge-reranker-v2-m3" # Only used if top_n is provided

    @model_validator(mode="after")
    def _validate(cls, values):
        """
        Validate that 'type' is valid for the given 'namespace'.
        """
        return validate_type_vs_namespace(values)



def validate_type_vs_namespace(model):
    """
    Validation method that ensures the 'type' field is valid for the given 'namespace'.
    """
    valid_types = {
        "main": {"deal", "company", "contact"},
        "all_meetings": {"full_summary", "short_summary", "meeting_chunk"}
    }

    ns = model.namespace
    t = model.type

    if t is not None and t not in valid_types.get(ns, set()):
        raise ValueError(
            f"Invalid type '{t}' for namespace '{ns}'. Must be one of {valid_types[ns]}"
        )

    return model