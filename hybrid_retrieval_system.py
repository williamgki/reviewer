#!/usr/bin/env python3
"""
Hybrid Retrieval System for LessWrong Contextual Chunks
Combines BGE-M3 dense embeddings + BM25 lexical search + reciprocal rank fusion.

The module attempts to download required NLTK resources (``punkt`` and
``stopwords``) at import time. In offline environments where these resources
cannot be downloaded, it logs a warning and falls back to a minimal whitespace
tokenizer without stopword removal to avoid blocking execution.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML/IR libraries
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def _ensure_nltk_resource(resource: str, path: str) -> bool:
    """Ensure an NLTK resource is available, downloading quietly if possible."""
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        nltk.download(resource, quiet=True, raise_on_error=False)
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            return False


PUNKT_AVAILABLE = _ensure_nltk_resource('punkt', 'tokenizers/punkt')
STOPWORDS_AVAILABLE = _ensure_nltk_resource('stopwords', 'corpora/stopwords')

if not PUNKT_AVAILABLE:
    logger.warning("NLTK 'punkt' tokenizer not available; falling back to basic split tokenization.")
if not STOPWORDS_AVAILABLE:
    logger.warning("NLTK 'stopwords' corpus not available; stopword filtering disabled.")


def tokenize(text: str) -> List[str]:
    """Tokenize text using NLTK if available, else a simple split."""
    return word_tokenize(text) if PUNKT_AVAILABLE else text.split()

@dataclass
class SearchResult:
    """Individual search result with scores"""
    chunk_id: str
    doc_id: str
    content: str
    section_path: str
    summary_header: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = None

class HybridRetrievalSystem:
    """
    Production hybrid retrieval system combining:
    - Dense: BGE-M3 embeddings with FAISS HNSW
    - Sparse: BM25 lexical search
    - Fusion: Reciprocal rank fusion + weighted scoring
    - Re-ranking: Cross-encoder for top-K refinement
    """
    
    def __init__(self, 
                 chunks_path: str = "chunked_corpus/contextual_chunks_complete.parquet",
                 index_dir: str = "retrieval_indexes",
                 embedding_model: str = "BAAI/bge-m3",
                 dense_weight: float = None,
                 bm25_weight: float = None,
                 link_weight: float = None,
                 config: dict = None):
        
        self.chunks_path = chunks_path
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Scoring weights - use config if provided
        if config and "retrieval" in config:
            rw = config["retrieval"]
            dense_weight = rw.get("dense_weight", dense_weight or 0.55)
            bm25_weight = rw.get("bm25_weight", bm25_weight or 0.35)
            link_weight = rw.get("link_weight", link_weight or 0.10)
        
        self.dense_weight = dense_weight or 0.55
        self.bm25_weight = bm25_weight or 0.35
        self.link_weight = link_weight or 0.10
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.bm25_index = None
        self.chunks_df = None
        self.chunk_texts = None
        self.stop_words = set(stopwords.words('english')) if STOPWORDS_AVAILABLE else set()
        self.building_index = False  # Track when building indexes to use CPU
        
        logger.info(f"🔍 Initializing Hybrid Retrieval System")
        logger.info(f"   Dense weight: {dense_weight}")
        logger.info(f"   BM25 weight: {bm25_weight}")
        logger.info(f"   Link weight: {link_weight}")
        
    def load_chunks(self) -> pd.DataFrame:
        """Load contextual chunks from parquet"""
        logger.info(f"📚 Loading chunks from {self.chunks_path}")
        
        self.chunks_df = pd.read_parquet(self.chunks_path)
        logger.info(f"✅ Loaded {len(self.chunks_df)} contextual chunks")
        
        # Create searchable text (content + section_path + summary)
        self.chunk_texts = []
        for _, row in self.chunks_df.iterrows():
            searchable_text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            self.chunk_texts.append(searchable_text)
            
        return self.chunks_df
        
    def initialize_embedding_model(self, model_name: str = "BAAI/bge-m3"):
        """Initialize BGE-M3 embedding model"""
        logger.info(f"🤖 Loading BGE-M3 model: {model_name}")
        
        # Force CPU for bulk index builds to avoid MPS pitfalls
        device_for_index = "cpu" if self.building_index else "auto"
        self.embedding_model = SentenceTransformer(model_name, device=device_for_index)
        
        # Cap sequence length so we never feed giant tensors to MPS/CPU
        try:
            max_seq_length = getattr(self, "max_seq_length", 1024)
            self.embedding_model.max_seq_length = min(max_seq_length, 2048)
        except Exception:
            pass
            
        logger.info(f"✅ BGE-M3 loaded - Max tokens: {self.embedding_model.max_seq_length}")
        logger.info(f"   Device: {device_for_index} (building_index={self.building_index})")
        logger.info(f"   Embedding dim: {self.embedding_model.get_sentence_embedding_dimension()}")
        
    def build_dense_index(self, force_rebuild: bool = False):
        """Build FAISS HNSW index for dense retrieval"""
        faiss_path = self.index_dir / "bge_m3_faiss.index"
        embeddings_path = self.index_dir / "bge_m3_embeddings.pkl"
        
        if faiss_path.exists() and embeddings_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing FAISS index...")
            self.faiss_index = faiss.read_index(str(faiss_path))
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
            return
            
        logger.info("🔨 Building FAISS dense index with BGE-M3...")
        
        if self.embedding_model is None:
            self.initialize_embedding_model()
            
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.chunk_texts), batch_size):
            batch_texts = self.chunk_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
            
            if i % 1000 == 0:
                logger.info(f"   Processed {i + len(batch_texts)}/{len(self.chunk_texts)} chunks")
                
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings)
        logger.info(f"✅ Generated embeddings: {self.embeddings.shape}")
        
        # Build FAISS HNSW index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
        self.faiss_index.hnsw.ef_construction = 200  # Higher quality construction
        self.faiss_index.hnsw.ef_search = 100  # Search quality
        
        # Add vectors to index
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        # Save index and embeddings
        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
            
        logger.info(f"✅ FAISS index built and saved: {self.faiss_index.ntotal} vectors")
        
    def build_bm25_index(self, force_rebuild: bool = False):
        """Build BM25 index for lexical search"""
        bm25_path = self.index_dir / "bm25_index.pkl"
        
        if bm25_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing BM25 index...")
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            logger.info(f"✅ BM25 index loaded: {len(self.bm25_index.doc_freqs)} documents")
            return
            
        logger.info("🔨 Building BM25 lexical index...")
        
        # Tokenize texts for BM25
        tokenized_texts = []
        for text in self.chunk_texts:
            # Simple tokenization and stopword removal
            tokens = tokenize(text.lower())
            tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            tokenized_texts.append(tokens)
            
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        # Save BM25 index
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
            
        logger.info(f"✅ BM25 index built and saved: {len(tokenized_texts)} documents")
        
    def build_indexes(self, force_rebuild: bool = False):
        """Build all indexes (dense + sparse)"""
        logger.info("🏗️  Building hybrid retrieval indexes...")
        
        # Set building flag for CPU-only embedding during index builds
        self.building_index = True
        
        try:
            # Load data first
            if self.chunks_df is None:
                self.load_chunks()
                
            # Initialize embedding model
            if self.embedding_model is None:
                self.initialize_embedding_model()
                
            # Build indexes
            self.build_dense_index(force_rebuild)
            self.build_bm25_index(force_rebuild)
            
            logger.info("✅ All indexes built successfully!")
        finally:
            # Reset building flag
            self.building_index = False
        
    def search_dense(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Dense search using BGE-M3 + FAISS"""
        if self.faiss_index is None or self.embedding_model is None:
            raise ValueError("Dense index not built. Call build_indexes() first.")
            
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search FAISS
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # Return (index, score) pairs
        results = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
        return results
        
    def search_bm25(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Sparse search using BM25"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_indexes() first.")
            
        # Tokenize query
        query_tokens = tokenize(query.lower())
        query_tokens = [token for token in query_tokens if token.isalnum() and token not in self.stop_words]
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices and scores
        top_indices = np.argsort(scores)[-k:][::-1]  # Reverse for descending
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results
        
    def reciprocal_rank_fusion(self, 
                             dense_results: List[Tuple[int, float]], 
                             bm25_results: List[Tuple[int, float]],
                             k: int = 60) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF) to combine rankings
        RRF score = 1/(k + rank) for each list
        """
        rrf_scores = {}
        
        # Add dense results
        for rank, (doc_id, score) in enumerate(dense_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Add BM25 results
        for rank, (doc_id, score) in enumerate(bm25_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        return rrf_scores
        
    def hybrid_search(self, 
                     query: str, 
                     k: int = 20,
                     dense_k: int = 200,
                     bm25_k: int = 200,
                     use_rrf: bool = True) -> List[SearchResult]:
        """
        Hybrid search combining dense + sparse with fusion
        """
        logger.info(f"🔍 Hybrid search: '{query[:50]}...'")
        
        # Get individual search results
        dense_results = self.search_dense(query, k=dense_k)
        bm25_results = self.search_bm25(query, k=bm25_k)
        
        if use_rrf:
            # Use Reciprocal Rank Fusion
            rrf_scores = self.reciprocal_rank_fusion(dense_results, bm25_results)
            
            # Sort by RRF score
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Create SearchResult objects
            search_results = []
            for rank, (doc_idx, rrf_score) in enumerate(sorted_results):
                row = self.chunks_df.iloc[doc_idx]
                
                # Get original scores
                dense_score = next((score for idx, score in dense_results if idx == doc_idx), 0.0)
                bm25_score = next((score for idx, score in bm25_results if idx == doc_idx), 0.0)
                
                result = SearchResult(
                    chunk_id=row['chunk_id'],
                    doc_id=row['doc_id'],
                    content=row['content'][:500] + "..." if len(row['content']) > 500 else row['content'],
                    section_path=row.get('section_path', ''),
                    summary_header=row.get('summary_header', ''),
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    combined_score=rrf_score,
                    rank=rank + 1,
                    metadata={"fusion_method": "rrf"}
                )
                search_results.append(result)
                
        else:
            # Use weighted linear combination
            combined_scores = {}
            
            # Normalize scores
            dense_scores = {idx: score for idx, score in dense_results}
            bm25_scores = {idx: score for idx, score in bm25_results}
            
            # Min-max normalization
            if dense_scores:
                max_dense = max(dense_scores.values())
                min_dense = min(dense_scores.values())
                if max_dense > min_dense:
                    dense_scores = {idx: (score - min_dense) / (max_dense - min_dense) 
                                  for idx, score in dense_scores.items()}
                    
            if bm25_scores:
                max_bm25 = max(bm25_scores.values())
                min_bm25 = min(bm25_scores.values())
                if max_bm25 > min_bm25:
                    bm25_scores = {idx: (score - min_bm25) / (max_bm25 - min_bm25) 
                                 for idx, score in bm25_scores.items()}
            
            # Combine scores
            all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
            for doc_idx in all_indices:
                dense_score = dense_scores.get(doc_idx, 0.0)
                bm25_score = bm25_scores.get(doc_idx, 0.0)
                combined_scores[doc_idx] = (self.dense_weight * dense_score + 
                                          self.bm25_weight * bm25_score)
                
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Create SearchResult objects
            search_results = []
            for rank, (doc_idx, combined_score) in enumerate(sorted_results):
                row = self.chunks_df.iloc[doc_idx]
                
                result = SearchResult(
                    chunk_id=row['chunk_id'],
                    doc_id=row['doc_id'],
                    content=row['content'][:500] + "..." if len(row['content']) > 500 else row['content'],
                    section_path=row.get('section_path', ''),
                    summary_header=row.get('summary_header', ''),
                    dense_score=dense_scores.get(doc_idx, 0.0),
                    bm25_score=bm25_scores.get(doc_idx, 0.0),
                    combined_score=combined_score,
                    rank=rank + 1,
                    metadata={"fusion_method": "weighted_linear"}
                )
                search_results.append(result)
        
        logger.info(f"✅ Found {len(search_results)} results")
        return search_results

def main():
    """Test the hybrid retrieval system"""
    
    # Initialize system
    retrieval_system = HybridRetrievalSystem(
        chunks_path="chunked_corpus/contextual_chunks_complete.parquet",
        dense_weight=0.55,
        bm25_weight=0.35,
        link_weight=0.10
    )
    
    # Build indexes
    retrieval_system.build_indexes(force_rebuild=False)
    
    # Test queries
    test_queries = [
        "AI alignment and safety research",
        "machine learning interpretability",
        "rationality and decision making",
        "artificial general intelligence risks"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = retrieval_system.hybrid_search(query, k=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.chunk_id[:12]}...] Score: {result.combined_score:.4f}")
            print(f"   Section: {result.section_path}")
            print(f"   Summary: {result.summary_header}")
            print(f"   Content: {result.content}")
            print(f"   Dense: {result.dense_score:.3f}, BM25: {result.bm25_score:.3f}")

if __name__ == "__main__":
    main()