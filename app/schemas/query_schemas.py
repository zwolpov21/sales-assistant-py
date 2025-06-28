from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Optional, Literal, Annotated

"""
Namespace Enum for Pinecone query namespaces
"""
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


""""
* metadata_filter -> follows Pinecone's filter syntax
* See https://docs.pinecone.io/guides/index-data/indexing-overview#metadata
    * Includes details on valid formats, all supported operators
* For singular metadata filters, use a dict with a single key-value pair. Example:

    ```metadata_filter = {"type": {"$eq": "deal"}}```

* For multiple metadata filters, use a dict with multiple key-value pairs. Example:

    ```metadata_filter = {
            "$and": [
                {"client_company": {"$eq": "1857 Advisors Group"}},
                {"type": {"$eq": "meeting_chunk"}}
            ]
        }
    ```
"""

"""
Request model for /query/dense_rerank
"""
class DenseRerankRequest(BaseModel):
    query: str
    top_k: int = 100
    top_n: int = 20
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    metadata_filter: Optional[dict] = None  # Optional metadata filter
    rerank_model: Optional[str] = "bge-reranker-v2-m3"

"""
Request model for /query/dense
"""
class DenseQueryRequest(BaseModel):
    query: str
    top_k: int = 20
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    metadata_filter: Optional[dict] = None  # Optional metadata filter

"""
Request model for /query/dense_multi_retrieve

Supports multiple queries in a single request.
Optionally rerank results by including a top_n parameter.
Optionally filter by metadata
"""
class DenseMultiQueryRequest(BaseModel):
    main_query: str
    num_queries: int = 3
    top_k: int = 100 # For each query!
    top_n: Optional[int] = 20 # For reranking each query's results, if None, reranks at end only
    namespace: Annotated[Namespace, Literal["main", "all_meetings"]]
    metadata_filter: Optional[dict] = None  # Optional metadata filter
    rerank_model: Optional[str] = "bge-reranker-v2-m3" # Only used if top_n is provided
