from fastapi import APIRouter, HTTPException
from app.services.openai_service import OpenAIService
from app.services.pinecone_service import PineconeService
import os
from pydantic import BaseModel
from app.schemas import query_schemas, openai_schemas
from app.services import llm_prompts
import logging

# router
router = APIRouter()

# logging
logger = logging.getLogger(__name__)

# initialize services
openai_client = OpenAIService(api_key=os.getenv("OPENAI_KEY"))
pinecone_client = PineconeService(api_key=os.getenv("PINECONE_KEY"))


@router.post("/query/dense_rerank")
async def query_dense_rerank(request_body: query_schemas.DenseRerankRequest):
    """
    Queries the dense Pinecone index and reranks the results.

    Supports optional metadata filtering by 'type'.
    Valid 'type' values
    - For the 'main' namespace: 'deal', 'company', 'contact'
    - For the 'all_meetings' namespace: 'full_summary', 'short_summary', 'meeting_chunk'

    Uses openai and pinecone services to handle embeddings,
    querying, reranking.
    """

    """
    1. Get dense embedding of query via openai
    """
    dense_embedding: list[float] = openai_client.get_openai_embeddings(
        query=request_body.query
    )["values"]

    """
    2. Query dense index with dense embedding
       Note - metadata_filter is optional, if None, no filtering will be applied
    """
    dense_results = pinecone_client.query_dense_index(
        embedding=dense_embedding,
        namespace=request_body.namespace,
        top_k=request_body.top_k,
        metadata_filter=request_body.metadata_filter,
    )

    """
    3. Rerank dense results
    """
    reranked_dense_results = pinecone_client.rerank_results(
        query=request_body.query,
        top_k_matches=dense_results,
        top_n=request_body.top_n,
        model=request_body.rerank_model
    )

    """
    4. Return reranked results
    """
    return {"hits": reranked_dense_results}


@router.post("/query/dense")
async def query_dense(request_body: query_schemas.DenseQueryRequest):
    """
    Queries the dense Pinecone index using the provided dense embedding.
    Simply retrieval, does not rerank results.
    
    Supports optional metadata filtering by different fields. 
    See `query_schemas` for details.
    """
    
    # Get dense embedding of query via OpenAI
    dense_embedding = openai_client.get_openai_embeddings(
        query=request_body.query
    )["values"]

    # Query dense index with dense embedding
    # Note - metadata_filter is optional, if None, no filtering will be applied
    try:
        dense_results = pinecone_client.query_dense_index(
            embedding=dense_embedding,
            namespace=request_body.namespace,
            metadata_filter=request_body.metadata_filter, 
            top_k=request_body.top_k
        )
    except Exception as e:
        logger.error(f"Error querying dense index: {e}")
        raise HTTPException(status_code=400, detail=f"Error querying dense index: {e}")

    return {"matches": dense_results}


@router.post("/query/dense_multi_retrieve")
async def query_dense_multi_retrieve(request_body: query_schemas.DenseMultiQueryRequest):
    """
    Supports multiple queries in a single request.
    - Optionally rerank results by including a top_n parameter.
    - Optionally filter by data 'type'.
    
    Request body params:
    - main_query: The main query string
    - num_queries: The number of queries to generate
    - top_k: The number of top results to retrieve for each query
    - top_n: The number of top results to rerank for each query (optional)
    - namespace: The namespace to query
    - type: The type of data to filter by (optional)
    - rerank_model: The model to use for reranking (default: "bge-reranker-v2-m3") (optional)

    Generates num_queries variations of the main_query -> retrieves top_k results for each ->
    deduplicate by ID -> split unique results into batches of 100 -> rerank each batch if top_n is provided
    -> combine all reranked batches -> globally rerank all results.
    """


    """
    0. Generate num_queries variations of the main_query
    """
    queries: list[str] = [request_body.main_query]
    queries += openai_client.get_chat_completion(
        messages= 
                {
                    "system": llm_prompts.GENERATE_QUERIES_SYS_PROMPT,
                    "user": llm_prompts.GENERATE_QUERIES_USER_PROMPT.format(
                        num_queries=request_body.num_queries,
                        main_query=request_body.main_query
                    )
                },
        model="gpt-4.1-mini",
        response_model=openai_schemas.GenerateQueriesReturnFormat
    ).get("output", []).queries
    logger.info(f"Generated queries: {queries}")


    """
    1. For each query, get dense embedding via OpenAI
    """
    dense_embeddings: list[list[float]] = []
    for query in queries:
        try:
            dense_embedding = openai_client.get_openai_embeddings(
                query=query
            )["values"]
            dense_embeddings.append(dense_embedding)
        except Exception as e:
            logger.error(f"Error embedding query '{query}': {e}")
            raise HTTPException(status_code=500, detail=f"Error embedding query '{query}': {e}")

    logger.info(f"Generated dense embeddings for {len(dense_embeddings)} queries.")
    
    """
    2. For each dense embedding, query dense index
    """
    top_k_dense_results: list[list[dict]] = []
    for dense_embedding in dense_embeddings:
        try:
            dense_results = pinecone_client.query_dense_index(
                embedding=dense_embedding,
                namespace=request_body.namespace,
                top_k=request_body.top_k,
                metadata_filter=request_body.metadata_filter,  # Optional metadata filter
            )
            top_k_dense_results.append(dense_results)
        except Exception as e:
            logger.error(f"Error querying dense index for embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Error querying dense index for embedding: {e}")
        
    logger.info(f"Queried dense index for {len(top_k_dense_results)} queries, each with top_k={request_body.top_k} results.")

    """
    3. Combine all results, deduplicate by ID
    """
    unique_results_dict: dict[str, dict] = {}
    for result_list in top_k_dense_results:
        for result in result_list:
            unique_results_dict[result['id']] = result

    unique_results: list[dict] = list(unique_results_dict.values())
    logger.info(f"Combined results, deduplicated by ID, total unique results: {len(unique_results)}")

    """
    4. Split unique results into batches of 100
       (most reranker models can only handle up to 100 results at a time)

       If top_n is NOT provided, we can skip this step, return raw unique results.
    """

    # Skip if top_n not passed in request body
    if request_body.top_n is None:
        return {"hits": unique_results}

    batches: list[list[dict]] = []
    batch_size = 100
    for i in range(0, len(unique_results), batch_size):
        batch = unique_results[i:i + batch_size]
        batches.append(batch)
    logger.info(f"Split unique results into {len(batches)} batches of size {batch_size}.")

    """
    5. Rerank each batch of results.
      
       top_n for this step is determined by the number of batches,
       because we need to ensure the final reranking step
       reranks AT MOST 100 results.
    """
    local_top_n: int = 100//len(batches)
    reranked_batches: list[list[dict]] = []
    for batch in batches:
        try:
            reranked_batch = pinecone_client.rerank_results(
                query=request_body.main_query,
                top_k_matches=batch,
                top_n=local_top_n,  # Rerank each batch to the determined local_top_n
                model=request_body.rerank_model
            )
            reranked_batches.append(reranked_batch)
        except Exception as e:
            logger.error(f"Error reranking batch: {e}")
            raise HTTPException(status_code=500, detail=f"Error reranking batch: {e}")
        
    logger.info(f"Reranked {len(reranked_batches)} batches of results, each with local_top_n={local_top_n}.")
    
    """
    6. Combine all reranked batches into a single list
       and globally rerank the results to get the final top_n results.
    """
    combined_reranked_results: list[dict] = []
    for reranked_batch in reranked_batches:
        combined_reranked_results.extend(reranked_batch)
    logger.info(f"Combined reranked results before global rerank: {len(combined_reranked_results)} results")

    # Warn if combined results exceed 100
    if(len(combined_reranked_results) > 100):
        logger.warning(f"Combined reranked results exceed 100, truncating to 100 results.")
        combined_reranked_results = combined_reranked_results[:100]

    try:
        global_reranked_results: list[dict] = pinecone_client.rerank_results(
            query=request_body.main_query,
            top_k_matches=combined_reranked_results,
            top_n=request_body.top_n,  # Final top_n results
            model=request_body.rerank_model
        )
    except Exception as e:
        logger.error(f"Error globally reranking results: {e}")
        raise HTTPException(status_code=500, detail=f"Error globally reranking results: {e}")
    logger.info(f"Globally reranked results, final top_n={len(global_reranked_results)}.")

    return {"reranked_hits": global_reranked_results}




# @router.post("/query/hybrid_mmr_retrieve")
# async def hybrid_query_mmr(request_body: query_schemas.HybridMMRRequest):

#     # 0. Embed the query – including both
#     #    dense and sparse embeddings

#     # Gets dense embedding of query
#     # Model: text-embedding-3-small
#     # Returns: list of floats (dense vector)
#     openai_response = openai_client.embeddings.create(
#         input=request_body.query,
#         model="text-embedding-3-small"
#     )
#     dense_embedding = openai_response.data[0].embedding

#     # Gets sparse embedding of query
#     # Model: pinecone-sparse-english-v0
#     # Returns: dictionary ("indices" and "values")
#     pinecone_response = pc.inference.embed(
#         model=EmbedModel.Pinecone_Sparse_English_V0,
#         inputs=[request_body.query],
#         parameters={
#             "input_type": "query",
#             "truncate": "END"
#         }
#     )
#     # print(pinecone_response)
#     sparse_embedding_indices = pinecone_response.data[0].get('sparse_indices')
#     sparse_embedding_values = pinecone_response.data[0].get('sparse_values')
    
#     # 1. Query for top_k (indicated below in method parameters)
#     #    records from dense index
#     dense_results = dense_index.query(
#         vector=dense_embedding,
#         top_k=100,
#         include_metadata=True,
#         include_values=True,  # include values for MMR
#         namespace=request_body.namespace
#     )["matches"]

#     # 2. Query for top_k (indicated below in method parameters)
#     #    records from sparse index
#     sparse_results = sparse_index.query(
#         sparse_vector={
#             "indices": sparse_embedding_indices,
#             "values": sparse_embedding_values
#         },
#         top_k=30,
#         include_metadata=True,
#         namespace=request_body.namespace
#     )["matches"]

#     # 3. ***Run mmr individually on dense results ONLY***
#     #    Why dense only? –> redundancy only becomes an issue
#     #    when user prompts require wide context coverage (the
#     #    sparse retrieval will rarely, if at all, have this issue,
#     #    as it only retrieves based off of keyword "matches").

#     # 3.1 –> Convert dense results to langchain Document objects
#     #        and Numpy embeddings. 
#     #        LangChain's maximum_marginal_relevance method takes
#     #        langchain Document objects and Numpy embeddings as parameters.
#     dense_docs: list[Document] = []
#     dense_embeddings: list[np.ndarray] = []
#     for result in dense_results:
#         # Extract text from the result
#         # –> If the result's "type" is "full_summary" or "short_summary",
#         #   then use the "summary_text" field for the Document text.
#         # –> If the result's "type" is "meeting_chunk", use "chunk_text"
#         text = result['metadata'].get("summary", "") or result['metadata'].get("chunk_text", "")

#         # 'id' is not listed in metadata –> add to metadata for
#         # deduplication purposes
#         metadata = result['metadata']
#         metadata['id'] = result.get('id', None)  # add id for deduplication 

#         dense_docs.append(Document(page_content=text, metadata=metadata))
#         dense_embeddings.append(np.array(result['values'], dtype=np.float32))
    
#     # 3.2 –> Convert query dense_embeddings to a Numpy array
#     np_query_dense_embedding = np.array(dense_embedding, dtype=np.float32)

#     # 3.3 –> Run MMR on the dense results
#     #        Note that lambda_mult is a float between 0 and 1 (passed as parameter),
#     #       and top_k is an integer (passed as parameter).
#     mmr_results = maximal_marginal_relevance(
#         query_embedding=np_query_dense_embedding,
#         embedding_list= np.array(dense_embeddings),
#         lambda_mult=request_body.lambda_mult,
#         k=request_body.mmr_top_k 
#     )
#     dense_mmr_docs = [dense_docs[i] for i in mmr_results]

#     # 4. Merge dense MMR results with sparse results, deduplicate

#     # 4.1 –> Convert sparse results to langchain Document objects
#     #        for easier merging and deduplication.
#     sparse_docs: list[Document] = []
#     for result in sparse_results:
#         # Results can be of "type" = "meeting_chunk" or "full_summary" or "short_summary"
#         text = result.get('metadata', {}).get('chunk_text', "") or result.get('metadata', {}).get('summary', "")

#         # 'id' is not listed in metadata –> add to metadata
#         # for deduplication purposes
#         metadata = result.get('metadata', {})
#         metadata['id'] = result.get('id', None)
#         sparse_docs.append(Document(page_content=text, metadata=metadata))
    

#     # 4.2 –> Deduplicate the combined results into list 
#     #        of Langchain documents
#     final_docs: list[Document] = []
#     ids = set()
#     for doc in dense_mmr_docs + sparse_docs:
#         doc_id = doc.metadata.get("id")
#         print("Doc ID: ", doc_id)
#         if doc_id not in ids:
#             final_docs.append(doc.metadata)
#             ids.add(doc_id)

#     return {"matches": final_docs}





