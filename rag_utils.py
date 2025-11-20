"""
RAG Retrieval Optimization Tool
Includes optimization strategies such as vector cache, dynamic k-value adjustment, and relevance filtering.
"""

import os
from functools import lru_cache
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

class OptimizedRAG:
    """Optimized RAG Retriever"""
    
    def __init__(self, persist_directory, dashscope_api_key):
        self.persist_directory = persist_directory
        self.dashscope_api_key = dashscope_api_key
        self._vectordb = None
        
    @property
    def vectordb(self):
        """Lazy-loading and caching vector databases"""
        if self._vectordb is None:
            print(f"[RAG] Load the vector database: {self.persist_directory}")
            embeddings = DashScopeEmbeddings(
                model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3"),
                dashscope_api_key=self.dashscope_api_key
            )
            self._vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
                collection_name="zinos_petrel_knowledge"  # Maintain consistency with vectorized scripts
            )
            print(f"[RAG] ✅ The vector database has been loaded.")
        return self._vectordb
    
    def retrieve(self, query, k=None, fetch_k=None, lambda_mult=0.7, 
                 relevance_threshold=None):
        """
         Smart Document Retrieval
        
        Args:
            query: Search text
            k: Number of documents to return (None for auto-adjustment)
            fetch_k: MMR candidate pool size (None for auto-adjustment)
            lambda_mult: MMR diversity parameter (0-1; higher values prioritize relevance, lower values prioritize diversity)
            relevance_threshold: Relevance threshold (0-1; filters low-quality documents)
        
        Returns:
            list: Retrieved document list
        """
        # Dynamically adjust the k value (based on query complexity)
        if k is None:
            k = self._estimate_k(query)
        
        # Dynamic Adjustment of fetch_k
        if fetch_k is None:
            fetch_k = k * 3  # The candidate pool is three times the number of returns.
        
        print(f"[RAG] Search Parameters: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        # Retrieval Using MMR (Maximum Marginal Relevance)
        docs = self.vectordb.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        # Relevance Filtering (if a threshold is set)
        if relevance_threshold is not None:
            filtered_docs = self._filter_by_relevance(
                query, docs, threshold=relevance_threshold
            )
            print(f"[RAG] Relevance Filtering: {len(docs)} -> {len(filtered_docs)} document")
            return filtered_docs
        
        return docs
    
    def _estimate_k(self, query):
        """
        Estimating k Value Based on Query Complexity
        
        Simple heuristic rules:
        - Short queries (<20 words): k=2
        - Medium queries (20-50 words): k=3
        - Complex queries (>50 words): k=4
        """
        word_count = len(query.split())
        
        if word_count < 20:
            return 2
        elif word_count < 50:
            return 3
        else:
            return 4
    
    def _filter_by_relevance(self, query, docs, threshold=0.6):
        """
        Filter documents based on relevance scores
        
        Args:
            query: Search text
            docs: Document list
            threshold: Relevance threshold (0-1)
        
        Returns:
            list: Filtered documents
        """
        # Use similarity_search_with_score to obtain relevance scores
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, 
            k=len(docs)
        )
        
        # Filter documents below the threshold
        # Note: Smaller ChromaDB distances indicate higher relevance (L2 distance), requiring conversion to similarity scores
        filtered = [
            doc for doc, score in docs_with_scores 
            if score < (1 - threshold)  # Distance Threshold Conversion
        ]
        
        return filtered if filtered else docs[:1]  # Return at least one document.
    
    def get_stats(self):
        """Retrieve vector library statistics"""
        collection = self.vectordb._collection
        count = collection.count()
        return {
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "embedding_model": os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
        }


# Global RAG Instance Caching (Preventing Repeated Loading)
_rag_instances = {}

def get_rag_instance(persist_directory, dashscope_api_key):
    """
    Get RAG Instance (With Cache)
    
    Args:
        persist_directory: Vector store path
        dashscope_api_key: DashScope API Key
    
    Returns:
        OptimizedRAG: RAG instance
    """
    if persist_directory not in _rag_instances:
        _rag_instances[persist_directory] = OptimizedRAG(
            persist_directory, 
            dashscope_api_key
        )
    return _rag_instances[persist_directory]

