from pinecone import Pinecone, EmbedModel
import numpy as np


# Connect to both index's
# Dense index: calls-crm-data-testing
# Sparse index: crm-calls-data-sparse
dense_index_name: str = "calls-crm-data-testing"
sparse_index_name: str = "crm-calls-data-sparse"

dense_url: str = "https://calls-crm-data-testing-43u8icu.svc.aped-4627-b74a.pinecone.io"
sparse_url: str = "https://crm-calls-data-sparse-43u8icu.svc.aped-4627-b74a.pinecone.io"

class PineconeService:
    def __init__(self, api_key: str):
        self.client = Pinecone(api_key=api_key)
        self.dense_index = self.client.Index(host=dense_url)
        self.sparse_index = self.client.Index(host=sparse_url)

    def get_sparse_embeddings(self, query: str) -> list[float]:
        """
        Gets sparse embedding of inputted query.
        Wraps Pinecone's 'Pinecone_Sparse_English_V0' model.

        Returns: dictionary ("indices" and "values")
        1. "indices": list of integers
        2. "values": list of floats
        """
        response = self.client.inference.embed(
            model=EmbedModel.Pinecone_Sparse_English_V0,
            inputs=[query],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )
        sparse_embedding_indices = response.data[0].get('sparse_indices')
        sparse_embedding_values = response.data[0].get('sparse_values')

        return {
            "indices": sparse_embedding_indices,
            "values": sparse_embedding_values
        }
    
    def query_dense_index(
        self, 
        embedding: list[float], 
        namespace: str = "main", 
        top_k: int = 20, 
        type: str = None
    ) -> list[dict]:
        """
        Queries the dense Pinecone index using the provided dense embedding.

        Supports optional metadata filtering by 'type'.
        Valid 'type' values
        - For the 'main' namespace: 'deal', 'company', 'contact'
        - For the 'all_meetings' namespace: 'full_summary', 'short_summary', 'meeting_chunk'

        Returns: list of matches (dictionaries)
        """

        """
        1. If type is provided, filter by type
           Otherwise, no filter
        """
        if type is not None:
            valid_types_main = ['deal', 'company', 'contact']
            valid_types_all_meetings = ['full_summary', 'short_summary', 'meeting_chunk']

            # Raise value error if invalid value for 'type' is provided
            if namespace == "main" and type not in valid_types_main:
                raise ValueError(f"Invalid type for namespace 'main'. Valid types are: {valid_types_main}")
            elif namespace == "all_meetings" and type not in valid_types_all_meetings:
                raise ValueError(f"Invalid type for namespace 'all_meetings'. Valid types are: {valid_types_all_meetings}")

            # Query with metadata filter for 'type'
            response = self.dense_index.query(
                vector=embedding,
                namespace=namespace,
                top_k=top_k,
                filter={
                    "type": {"$eq": type}
                },
                include_metadata=True
            )

        else:
            response = self.dense_index.query(
                vector=embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
        
        cleaned_matches = [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in response.matches
        ]
        
        return cleaned_matches

    def query_sparse_index(
        self, 
        sparse_embedding: dict, 
        namespace: str = "main", 
        top_k: int = 20,
        type: str = None
    ) -> list[dict]:
        """
        Queries the sparse Pinecone index using the provided sparse embedding.
        
        sparse_embedding parameter: {"indices": list of indices, "values": list of values}
        top_k parameter: number of top results to return
        namespace parameter: namespace to query in Pinecone index

        Supports optional metadata filtering by 'type'.
        Valid 'type' values
        - For the 'main' namespace: 'deal', 'company', 'contact'
        - For the 'all_meetings' namespace: 'full_summary', 'short_summary', 'meeting_chunk'

        Returns: list of matches (dictionaries)
        """

        """
        1. If type is provided, filter by type
           Otherwise, no filter
        """
        if type is not None:
            valid_types_main = ['deal', 'company', 'contact']
            valid_types_all_meetings = ['full_summary', 'short_summary', 'meeting_chunk']

            # Raise value error if invalid value for 'type' is provided
            if namespace == "main" and type not in valid_types_main:
                raise ValueError(f"Invalid type for namespace 'main'. Valid types are: {valid_types_main}")
            elif namespace == "all_meetings" and type not in valid_types_all_meetings:
                raise ValueError(f"Invalid type for namespace 'all_meetings'. Valid types are: {valid_types_all_meetings}")

            # Query with metadata filter for 'type'
            response = self.sparse_index.query(
                sparse_vector=sparse_embedding,
                namespace=namespace,
                top_k=top_k,
                filter={
                    "type": {"$eq": type}
                },
                include_metadata=True
        )
            
        else:
            response = self.sparse_index.query(
                sparse_vector=sparse_embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
        )
            
        return response.matches

    def rerank_results(
        self, 
        query: str, 
        top_k_matches: list[dict], 
        top_n: int, 
        model: str = "bge-reranker-v2-m3"
    ) -> list[dict]:
        """
        Simple reranking function that returns the top_n matches from the provided list
        based on the user query.

        Returns: list of top_n matches
        """
        
        """
        1. Extract chunk text for each match and rebuild.
           Note - need to handle all types of possible outputs
        """
        documents = []

        for match in top_k_matches:
            if "chunk_text" in match.get("metadata", {}):
                documents.append({'id': match.get('id', ''), 'text': match.get('metadata', {}).get('chunk_text', '')})
            elif "summary" in match.get('metadata', {}):
                documents.append({'id': match.get('id', ''), 'text': match.get('metadata', {}).get("summary", '')})
            elif match.get('metadata', {}).get('type', '') == "deal":
                deal_text = f"Deal Name: {match.get('metadata', {}).get('deal_name', '')}"
                deal_text += f"\nCompany Name: {match.get('metadata', {}).get('company_name', '')}"
                deal_text += f"\nInternal Team Type: {match.get('metadata', {}).get('type_of_team', '')}"
                deal_text += f"\nDeal Stage: {match.get('metadata', {}).get('stage', '')}"
                deal_text += f"\nCreated At: {match.get('metadata', {}).get('created_at', '')}"
                deal_text += f"\nIndustry Vertical: {match.get('metadata', {}).get('vertical', '')}"

                documents.append({"id": match.get('id', ''), "text": deal_text})

        reranked_docs = self.client.inference.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True
        )

        """
        2. Extract the reranked results from the response.
        """
        matches: list[dict] = []
        for match in reranked_docs.data:
            # find the match in top_k_matches by id
            original_match = next((m for m in top_k_matches if m['id'] == match.document.id), None)
            if original_match:
                matches.append({
                    "index": match.index,
                    "score": match.score,
                    "id": match.document.id,
                    "text": match.document.text,
                    "metadata": original_match.get("metadata", {})
                })

        return matches


