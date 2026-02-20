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
                collection_name="mmf_zinospetrel_knowledge"  # Maintain consistency with vectorized scripts
            )
            print(f"[RAG] ✅ The vector database has been loaded.")
        return self._vectordb
    
    def _priority_score(self, doc):
        """Compute a simple priority-based score for reranking."""
        # priority is stored as int-like metadata
        try:
            priority = int(doc.metadata.get("priority", 0))
        except Exception:
            priority = 0

        # scope is now stored as a STRING (e.g. "specimen,mmf,funchal")
        scope = str(doc.metadata.get("scope", "")).lower()

        bonus = 0
        if "specimen" in scope or "mmf" in scope:
            bonus = 50

        return priority + bonus

    def _rerank_by_priority(self, docs):
        """Rerank retrieved docs by priority + scope bonus."""
        return sorted(docs, key=self._priority_score, reverse=True)

    
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
        
                # Retrieve a larger pool, then rerank and cut down to k
        pool_k = max(fetch_k, k * 5)

        docs = self.vectordb.max_marginal_relevance_search(
            query,
            k=pool_k,
            fetch_k=pool_k,
            lambda_mult=lambda_mult
        )

        docs = self._rerank_by_priority(docs)
        docs = docs[:k]
        
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
        Filter the *given docs* by relevance using similarity_search_with_score
        and keeping only those that appear in docs.
        """
        docs_with_scores = self.vectordb.similarity_search_with_score(query, k=max(len(docs), 1))

        # map content -> distance (rough match)
        score_map = {}
        for d, dist in docs_with_scores:
            score_map[d.page_content] = dist

        kept = []
        for d in docs:
            dist = score_map.get(d.page_content, None)
            if dist is None:
                kept.append(d)  # keep if unknown
                continue
            if dist < (1 - threshold):
                kept.append(d)

        return kept if kept else docs[:1]
    
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

