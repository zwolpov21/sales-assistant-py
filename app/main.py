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


# FastAPI application instance
app = FastAPI()

# Load env variables
load_dotenv()


client = OpenAI(api_key='OPENAI_KEY')
pc = Pinecone(api_key='PINECONE_KEY')

# Connect to both index's
# Dense index: calls-crm-data-testing
# Sparse index: crm-calls-data-sparse
dense_index_name: str = "calls-crm-data-testing"
sparse_index_name: str = "crm-calls-data-sparse"

dense_url: str = "https://calls-crm-data-testing-43u8icu.svc.aped-4627-b74a.pinecone.io"
sparse_url: str = "https://crm-calls-data-sparse-43u8icu.svc.aped-4627-b74a.pinecone.io"

"""
Request model for Hybrid MMR retrieval from Pinecone
"""
class HybridMMRRequest(BaseModel):
    """
    Request model for Hybrid MMRR.
    """
    query: str
    namespace: str = "main"
    lambda_mult: float = 0.5
    mmr_top_k: int = 20


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Vercel"}

@app.post("/hybrid_mmr_retrieve")
def hybrid_query_mmr(request_body: HybridMMRRequest):
    # Connect to both indexes (dense and sparse)
    dense_index = pc.Index(host=dense_url)
    sparse_index = pc.Index(host=sparse_url)

    # 0. Embed the query – including both
    #    dense and sparse embeddings

    # Gets dense embedding of query
    # Model: text-embedding-3-small
    # Returns: list of floats (dense vector)
    openai_response = client.embeddings.create(
        input=request_body.query,
        model="text-embedding-3-small"
    )
    dense_embedding = openai_response.data[0].embedding

    # Gets sparse embedding of query
    # Model: pinecone-sparse-english-v0
    # Returns: dictionary ("indices" and "values")
    pinecone_response = pc.inference.embed(
        model=EmbedModel.Pinecone_Sparse_English_V0,
        inputs=[request_body.query],
        parameters={
            "input_type": "query",
            "truncate": "END"
        }
    )
    # print(pinecone_response)
    sparse_embedding_indices = pinecone_response.data[0].get('sparse_indices')
    sparse_embedding_values = pinecone_response.data[0].get('sparse_values')
    
    # 1. Query for top_k (indicated below in method parameters)
    #    records from dense index
    dense_results = dense_index.query(
        vector=dense_embedding,
        top_k=100,
        include_metadata=True,
        include_values=True,  # include values for MMR
        namespace=request_body.namespace
    )["matches"]

    # 2. Query for top_k (indicated below in method parameters)
    #    records from sparse index
    sparse_results = sparse_index.query(
        sparse_vector={
            "indices": sparse_embedding_indices,
            "values": sparse_embedding_values
        },
        top_k=30,
        include_metadata=True,
        namespace=request_body.namespace
    )["matches"]

    # 3. ***Run mmr individually on dense results ONLY***
    #    Why dense only? –> redundancy only becomes an issue
    #    when user prompts require wide context coverage (the
    #    sparse retrieval will rarely, if at all, have this issue,
    #    as it only retrieves based off of keyword "matches").

    # 3.1 –> Convert dense results to langchain Document objects
    #        and Numpy embeddings. 
    #        LangChain's maximum_marginal_relevance method takes
    #        langchain Document objects and Numpy embeddings as parameters.
    dense_docs: list[Document] = []
    dense_embeddings: list[np.ndarray] = []
    for result in dense_results:
        # Extract text from the result
        # –> If the result's "type" is "full_summary" or "short_summary",
        #   then use the "summary_text" field for the Document text.
        # –> If the result's "type" is "meeting_chunk", use "chunk_text"
        text = result['metadata'].get("summary", "") or result['metadata'].get("chunk_text", "")

        # 'id' is not listed in metadata –> add to metadata for
        # deduplication purposes
        metadata = result['metadata']
        metadata['id'] = result.get('id', None)  # add id for deduplication 

        dense_docs.append(Document(page_content=text, metadata=metadata))
        dense_embeddings.append(np.array(result['values'], dtype=np.float32))
    
    # 3.2 –> Convert query dense_embeddings to a Numpy array
    np_query_dense_embedding = np.array(dense_embedding, dtype=np.float32)

    # 3.3 –> Run MMR on the dense results
    #        Note that lambad_mult is a float between 0 and 1 (passed as parameter),
    #       and top_k is an integer (passed as parameter).
    mmr_results = maximal_marginal_relevance(
        query_embedding=np_query_dense_embedding,
        embedding_list= np.array(dense_embeddings),
        lambda_mult=request_body.lambda_mult,
        k=request_body.mmr_top_k 
    )
    dense_mmr_docs = [dense_docs[i] for i in mmr_results]

    # 4. Merge dense MMR results with sparse results, deduplicate

    # 4.1 –> Convert sparse results to langchain Document objects
    #        for easier merging and deduplication.
    sparse_docs: list[Document] = []
    for result in sparse_results:
        # Results can be of "type" = "meeting_chunk" or "full_summary" or "short_summary"
        text = result.get('metadata', {}).get('chunk_text', "") or result.get('metadata', {}).get('summary', "")

        # 'id' is not listed in metadata –> add to metadata
        # for deduplication purposes
        metadata = result.get('metadata', {})
        metadata['id'] = result.get('id', None)
        sparse_docs.append(Document(page_content=text, metadata=metadata))
    

    # 4.2 –> Deduplicate the combined results into list 
    #        of Langchain documents
    final_docs: list[Document] = []
    ids = set()
    for doc in dense_mmr_docs + sparse_docs:
        doc_id = doc.metadata.get("id")
        print("Doc ID: ", doc_id)
        if doc_id not in ids:
            final_docs.append(doc.metadata)
            ids.add(doc_id)

    return {"matches": final_docs}