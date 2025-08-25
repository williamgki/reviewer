#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from hybrid_retrieval_system import HybridRetrievalSystem
except ImportError as e:
    # Capture the original ImportError so missing module information is visible
    print(f"Warning: Could not import HybridRetrievalSystem: {e}")
    raise ImportError(
        "HybridRetrievalSystem is required but could not be imported."
    ) from e

class SemanticCorpusConceptSampler:
    """
    Use the existing vector database to find semantically relevant corpus concepts
    for paper concepts, rather than extracting concepts from raw text.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.retrieval_system = None
        self.initialize_retrieval()
    
    def initialize_retrieval(self):
        """Initialize the hybrid retrieval system with existing indexes."""
        if HybridRetrievalSystem is None:
            print("❌ Hybrid retrieval system not available")
            return
            
        try:
            # Use the existing retrieval indexes
            index_dir = Path(__file__).parent.parent / "retrieval_indexes"
            corpus_dir = Path(__file__).parent.parent / "chunked_corpus"
            
            if not index_dir.exists():
                print(f"❌ Index directory not found: {index_dir}")
                return
                
            if not corpus_dir.exists():
                print(f"❌ Corpus directory not found: {corpus_dir}")
                return
                
            self.retrieval_system = HybridRetrievalSystem(
                index_dir=str(index_dir),
                chunks_path=str(corpus_dir / "contextual_chunks_complete.parquet"),
                config=self.config
            )
            
            # Build both dense and BM25 indexes (includes load_chunks and initialize_embedding_model)
            cfg_force = self.config.get("retrieval", {}).get("force_rebuild", False)
            self.retrieval_system.build_indexes(force_rebuild=cfg_force)
            
            # 🔧 CONFIG DIAGNOSTIC: Print config as received by sampler
            retrieval_config = self.config.get("retrieval", {})
            print(f"✅ Semantic corpus concept sampler initialized")
            print(f"🔧 CONFIG: top_k={retrieval_config.get('top_k', 10)}, "
                  f"min_semantic_score={retrieval_config.get('min_semantic_score', 0.1)}, "
                  f"dense_weight={retrieval_config.get('dense_weight', 0.65)}")
            
        except Exception as e:
            print(f"⚠️ Init failed once ({e}); retrying with force_rebuild=True")
            try:
                self.retrieval_system.build_indexes(force_rebuild=True)
                print("✅ Semantic corpus concept sampler initialized (after retry)")
            except Exception as e2:
                print(f"❌ Failed to initialize retrieval system after retry: {e2}")
                raise
    
    def find_relevant_corpus_concepts(self, paper_concepts: List[Dict[str, Any]], 
                                    concepts_per_paper_concept: int = 100) -> List[Dict[str, Any]]:
        """
        For each paper concept, use semantic search to find relevant corpus chunks,
        then extract high-quality concepts from those chunks.
        """
        if not self.retrieval_system:
            print("❌ No retrieval system available")
            return []
        
        print(f"🔍 Finding relevant corpus concepts for {len(paper_concepts)} paper concepts...")
        
        all_corpus_concepts = []
        
        for i, paper_concept in enumerate(paper_concepts):
            concept_name = paper_concept.get('concept', paper_concept.get('name', ''))
            concept_backpack = paper_concept.get('backpack', '')
            
            print(f"  {i+1}/{len(paper_concepts)}: '{concept_name}'")
            
            # Create search query from paper concept
            search_query = self._create_search_query(concept_name, concept_backpack)
            
            try:
                # 🔧 ENHANCED RETRIEVAL: Multiple queries + broader search + per-doc capping
                retrieval_config = self.config.get("retrieval", {})
                k_base = retrieval_config.get("enhanced_k", 20)  # Increased from 10
                top_n = retrieval_config.get("rank_based_top_n", 5)
                per_doc_cap = retrieval_config.get("per_doc_cap", 2)  # Max results per document
                
                print(f"    🔍 Enhanced search: k={k_base}, top_n={top_n}, per_doc_cap={per_doc_cap}")
                
                # Generate multiple query variants for better coverage
                query_variants = self._generate_query_variants(concept_name, concept_backpack)
                
                all_search_results = []
                for i, query in enumerate(query_variants):
                    print(f"    📝 Query {i+1}/{len(query_variants)}: '{query[:50]}...'")
                    
                    search_results = self.retrieval_system.hybrid_search(
                        query=query,
                        k=k_base,
                        dense_k=50,
                        bm25_k=50
                    )
                    all_search_results.extend(search_results)
                
                print(f"    ✅ Found {len(all_search_results)} total results from {len(query_variants)} queries")
                
                # Apply per-document capping to ensure variety
                capped_results = self._apply_per_doc_cap(all_search_results, per_doc_cap)
                print(f"    📊 After per-doc cap: {len(capped_results)} results from {len(set(r.doc_id for r in capped_results))} unique docs")
                
                # Extract concepts from search results
                corpus_concepts = self._extract_concepts_from_results(
                    capped_results, paper_concept, max_concepts=k_base
                )
                
                # Score distribution diagnostic
                if corpus_concepts:
                    scores = [c.get('combined_score', 0) for c in corpus_concepts]
                    print(f"    🔧 Score range: min={min(scores):.4f}, median={sorted(scores)[len(scores)//2]:.4f}, max={max(scores):.4f}")
                
                # Always use rank-based filtering for enhanced system
                corpus_concepts_sorted = sorted(corpus_concepts, key=lambda x: x.get('combined_score', 0), reverse=True)
                quality_concepts = corpus_concepts_sorted[:top_n]
                print(f"    🏆 Rank-based: Kept top {len(quality_concepts)} of {len(corpus_concepts)} (top_n={top_n})")
                
                all_corpus_concepts.extend(quality_concepts)
                
            except Exception as e:
                print(f"    ❌ Search failed for '{concept_name}': {e}")
                continue
        
        # Deduplicate and rank corpus concepts
        final_concepts = self._deduplicate_and_rank(all_corpus_concepts)
        
        print(f"✅ Found {len(final_concepts)} relevant corpus concepts")
        return final_concepts
    
    def _create_search_query(self, concept_name: str, concept_backpack: str) -> str:
        """Create a search query that will find semantically related content."""
        
        # Combine concept name with context from backpack
        query_parts = [concept_name]
        
        if concept_backpack:
            # Extract key terms from backpack
            important_words = []
            for word in concept_backpack.split():
                word_clean = word.lower().strip('.,!?;:"()[]{}')
                if (len(word_clean) > 4 and 
                    word_clean not in {'about', 'which', 'where', 'there', 'their', 'these', 'those'}):
                    important_words.append(word_clean)
            
            # Add most relevant backpack terms
            if important_words:
                query_parts.extend(important_words[:3])  # Top 3 terms
        
        return ' '.join(query_parts)
    
    def _generate_query_variants(self, concept_name: str, concept_backpack: str) -> List[str]:
        """Generate multiple query variants for better coverage."""
        variants = []
        
        # Query 1: Exact phrase
        base_query = self._create_search_query(concept_name, concept_backpack)
        variants.append(base_query)
        
        # Query 2: Paraphrase/expanded version
        if concept_backpack:
            # Extract key terms from backpack for expansion
            backpack_terms = []
            for word in concept_backpack.split():
                clean_word = word.lower().strip('.,!?;:"()[]{}')
                if len(clean_word) > 4 and clean_word not in {'about', 'which', 'where', 'there', 'their', 'these', 'those', 'would', 'could', 'should'}:
                    backpack_terms.append(clean_word)
            
            if backpack_terms:
                expanded_query = f"{concept_name} {' '.join(backpack_terms[:3])}"
                variants.append(expanded_query)
        
        # Query 3: Domain synonyms/aliases
        domain_synonyms = self._get_domain_synonyms(concept_name)
        if domain_synonyms:
            # Pick the best synonym and create focused query
            synonym_query = f"{domain_synonyms[0]} {concept_name.split()[-1] if len(concept_name.split()) > 1 else ''}".strip()
            variants.append(synonym_query)
        
        return variants
    
    def _get_domain_synonyms(self, concept_name: str) -> List[str]:
        """Get domain-specific synonyms for AI safety concepts."""
        concept_lower = concept_name.lower()
        
        synonym_map = {
            'truthfulness': ['factuality', 'honesty', 'calibration', 'truthful QA'],
            'ai truthfulness': ['factuality', 'hallucination rate', 'calibration error'],
            'honesty policy': ['truthfulness framework', 'deception prevention', 'honest AI'],
            'alignment': ['AI safety', 'value alignment', 'beneficial AI'],
            'ai alignment': ['AI safety', 'value alignment', 'beneficial AI', 'alignment problem'],
            'language models': ['LLMs', 'large language models', 'neural language models'],
            'truthfulness in language models': ['LLM factuality', 'model calibration', 'hallucination'],
            'deception': ['dishonesty', 'misinformation', 'sycophancy'],
            'specification gaming': ['reward hacking', 'Goodhart\'s law', 'mesa-optimization']
        }
        
        # Find matching synonyms
        for key, synonyms in synonym_map.items():
            if key in concept_lower or concept_lower in key:
                return synonyms
        
        # General fallbacks for common terms
        if 'truthful' in concept_lower:
            return ['factual', 'honest', 'calibrated']
        if 'alignment' in concept_lower:
            return ['safety', 'beneficial', 'aligned']
        if 'model' in concept_lower:
            return ['AI system', 'neural network', 'algorithm']
            
        return []
    
    def _apply_per_doc_cap(self, results: List[Any], per_doc_cap: int) -> List[Any]:
        """Apply per-document cap to ensure variety across sources."""
        from collections import defaultdict
        
        doc_counts = defaultdict(int)
        capped_results = []
        
        # Sort by combined score first
        sorted_results = sorted(results, key=lambda x: getattr(x, 'combined_score', 0), reverse=True)
        
        for result in sorted_results:
            doc_id = result.doc_id
            if doc_counts[doc_id] < per_doc_cap:
                capped_results.append(result)
                doc_counts[doc_id] += 1
        
        return capped_results
    
    def _extract_concepts_from_results(self, search_results: List[Any], 
                                     paper_concept: Dict[str, Any],
                                     max_concepts: int = 10) -> List[Dict[str, Any]]:
        """Extract high-quality concepts from search results."""
        concepts = []
        
        for result in search_results[:max_concepts]:
            # Extract metadata from search result (SearchResult dataclass)
            meta = result.metadata or {}
            concept_info = {
                'concept': result.doc_id,  # Use doc_id as concept initially
                'source': 'corpus_semantic_search',
                'snippet': result.content[:200],  # First 200 chars
                'doc_id': result.doc_id,
                'chunk_id': result.chunk_id,
                'section_path': result.section_path,
                'summary_header': result.summary_header,
                
                # Extract metadata with improved fallbacks
                'title': meta.get('title') or result.section_path or result.summary_header or f"Doc {result.doc_id[:8]}",
                'author': meta.get('author') or self._extract_domain_from_url(meta.get('url', '')),
                'authors': meta.get('authors') or [meta.get('author')] or [self._extract_domain_from_url(meta.get('url', ''))],
                'year': meta.get('year', 2020),
                'url': meta.get('url', ''),
                
                # Search scores
                'dense_score': result.dense_score,
                'bm25_score': result.bm25_score,
                'combined_score': result.combined_score or (result.dense_score * 0.7 + result.bm25_score * 0.3),
                
                # Context
                'paper_concept_source': paper_concept.get('concept', ''),
                'relevance_type': 'semantic_similarity',
                'metadata': meta  # Preserve original metadata
            }
            
            # Extract meaningful concept name (prefer section info over content)
            meaningful_concept = self._extract_meaningful_concept(result)
            if meaningful_concept:
                concept_info['concept'] = meaningful_concept
            
            concepts.append(concept_info)
        
        return concepts
    
    def _extract_meaningful_concept(self, result) -> Optional[str]:
        """Extract meaningful concept name prioritizing metadata over content."""
        import re
        
        # Priority 1: Use section_path or summary_header if meaningful
        if hasattr(result, 'section_path') and result.section_path:
            section_clean = result.section_path.strip().replace('/', ' > ')
            if len(section_clean) > 3 and not section_clean.isdigit():
                return section_clean[:100]  # Reasonable length
                
        if hasattr(result, 'summary_header') and result.summary_header:
            header_clean = result.summary_header.strip()
            if len(header_clean) > 3 and not header_clean.isdigit():
                return header_clean[:100]
        
        # Priority 2: Extract from metadata if available
        meta = getattr(result, 'metadata', {}) or {}
        if meta.get('title') and len(meta['title']) > 3:
            return meta['title'][:100]
            
        # Priority 3: Technical terms from content
        content = result.content
        technical_patterns = [
            r'\b(?:reinforcement learning|deep learning|machine learning|artificial intelligence)\b',
            r'\b(?:transformer|attention mechanism|neural network|language model)\b', 
            r'\b(?:ai alignment|ai safety|mechanistic interpretability|reward hacking)\b',
            r'\b(?:rlhf|fine.?tuning|few.?shot|constitutional ai)\b',
            r'\b(?:autoencoder|embedding|activation|gradient descent)\b'
        ]
        
        for pattern in technical_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group().lower()
        
        # Priority 4: Clean snippet from content (avoid hashes)
        clean_snippet = re.sub(r'\b[a-f0-9]{32,}\b', '', content[:100])  # Remove hashes
        clean_snippet = re.sub(r'\s+', ' ', clean_snippet).strip()
        if len(clean_snippet) > 10:
            return clean_snippet
        
        return None
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL for fallback author."""
        if not url:
            return "Unknown"
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if domain:
                # Clean up domain (remove www, take main part)
                domain = domain.replace('www.', '')
                return domain.split('.')[0].title()  # e.g., "lesswrong" -> "Lesswrong"
        except:
            pass
            
        return "Unknown"
    
    def _deduplicate_and_rank(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank by relevance."""
        from collections import defaultdict
        
        # Group by concept name
        concept_groups = defaultdict(list)
        for concept in concepts:
            normalized_name = concept['concept'].lower().strip()
            concept_groups[normalized_name].append(concept)
        
        # Keep best example of each concept
        final_concepts = []
        for concept_name, group in concept_groups.items():
            # Sort by combined score (semantic + lexical)
            best_concept = max(group, key=lambda x: x.get('combined_score', 0))
            final_concepts.append(best_concept)
        
        # Sort by relevance and return
        final_concepts.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return final_concepts

# Integration function
def enhance_ddl_sampler_with_semantic_search(sampler_instance, config=None):
    """Enhance existing DDL sampler with semantic corpus concept discovery."""
    
    semantic_sampler = SemanticCorpusConceptSampler(config=config)
    
    # Add method to sampler instance
    def load_semantic_corpus_concepts(paper_concepts, target_concepts=1000):
        concepts = semantic_sampler.find_relevant_corpus_concepts(
            paper_concepts, 
            concepts_per_paper_concept=min(target_concepts // len(paper_concepts), 100)
        )
        return concepts[:target_concepts]
    
    sampler_instance.load_semantic_corpus_concepts = load_semantic_corpus_concepts
    return sampler_instance