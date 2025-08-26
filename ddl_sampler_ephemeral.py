#!/usr/bin/env python3

import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re


class EphemeralDDLSampler:
    """
    Ephemeral sampler for paper×corpus concept pairs.
    No persistence - all sampling happens within a single run.
    Uses importance sampling to find novel yet plausible connections.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.random = random.Random()  # For reproducible sampling
        
        # Diversity tracking (per-run only)
        self.used_authors = set()
        self.used_years = set()
        self.used_topics = set()
        self.sampled_pairs = set()
        
    def load_corpus_concepts(self, corpus_path: str, use_llm_extraction: bool = True) -> List[Dict[str, Any]]:
        """
        Load corpus concepts from chunked corpus data.
        Expected to extract key terms, entities, and technical concepts.
        """
        if use_llm_extraction:
            print("Using LLM-based corpus concept extraction...")
            try:
                from corpus_concept_extractor import CorpusConceptExtractor
                extractor = CorpusConceptExtractor()
                return extractor.extract_corpus_concepts(corpus_path, max_concepts=2000)
            except Exception as e:
                print(f"LLM corpus extraction failed: {e}")
                print("Falling back to regex-based extraction...")
        
        # Fallback to original method
        if os.path.isdir(corpus_path):
            import glob
            parquet_files = glob.glob(os.path.join(corpus_path, "*.parquet"))
            if parquet_files:
                return self._load_from_parquet(parquet_files[0])  # Load from first file for now
        
        if corpus_path.endswith('.parquet'):
            return self._load_from_parquet(corpus_path)
        else:
            return self._load_from_jsonl(corpus_path)
    
    def _load_from_parquet(self, parquet_path: str) -> List[Dict[str, Any]]:
        """Load corpus concepts from parquet file."""
        try:
            df = pd.read_parquet(parquet_path)
            concepts = []
            
            # Extract concepts from each chunk
            for _, row in df.iterrows():
                text = row.get('content', '')  # Use 'content' column from parquet
                author = row.get('authors', 'Unknown')  # Use 'authors' column
                year = 2020  # Default year since pub_date might be None
                if pd.notna(row.get('pub_date')):
                    try:
                        year = int(str(row['pub_date'])[:4])
                    except:
                        year = 2020
                title = row.get('doc_title', 'Untitled')
                doc_id = row.get('doc_id', 'unknown')
                
                # Extract key terms from text
                extracted_concepts = self._extract_corpus_concepts(text, author, year, title, doc_id)
                concepts.extend(extracted_concepts)
            
            return self._deduplicate_corpus_concepts(concepts)
            
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return []
    
    def _load_from_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load corpus concepts from JSONL file."""
        concepts = []
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'concept' in data and 'snippet' in data:
                            concepts.append(data)
        except Exception as e:
            print(f"Error loading JSONL: {e}")
        
        return concepts
    
    def _extract_corpus_concepts(self, text: str, author: str, year: int, 
                                title: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract concepts from a corpus text chunk."""
        concepts = []
        
        # Technical terms (regex patterns for common ML/AI concepts)
        tech_patterns = [
            r'\b(?:neural|deep|machine)\s+(?:network|learning|model)s?\b',
            r'\b(?:transformer|attention|embedding|gradient)s?\b',
            r'\b(?:reinforcement|supervised|unsupervised)\s+learning\b',
            r'\b(?:sparse|dense)\s+(?:autoencoder|representation)s?\b',
            r'\b(?:activation|feature|concept)\s+(?:patching|visualization|bottleneck)s?\b',
            r'\bbackpropagation\b', r'\boptimization\b', r'\bcalibration\b',
            r'\bhallucination\b', r'\balignment\b', r'\binterpretability\b'
        ]
        
        text_lower = text.lower()
        
        # Extract technical terms
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Get surrounding context (1-2 sentences)
                context_match = re.search(
                    rf'.{{0,80}}{re.escape(match)}.{{0,80}}', 
                    text, re.IGNORECASE
                )
                if context_match:
                    snippet = context_match.group().strip()
                    concepts.append({
                        'concept': match.lower(),
                        'source': 'corpus',
                        'snippet': snippet,
                        'author': author,
                        'year': year,
                        'title': title,
                        'doc_id': doc_id,
                        'embedding': None  # Will be computed later
                    })
        
        # Extract capitalized technical terms
        cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for term in cap_terms:
            term_lower = term.lower()
            if (len(term_lower) >= 4 and 
                not self._is_common_word(term_lower) and
                text_lower.count(term_lower) >= 1):
                
                context_match = re.search(
                    rf'.{{0,80}}{re.escape(term)}.{{0,80}}', 
                    text, re.IGNORECASE
                )
                if context_match:
                    snippet = context_match.group().strip()
                    concepts.append({
                        'concept': term_lower,
                        'source': 'corpus',
                        'snippet': snippet,
                        'author': author,
                        'year': year,
                        'title': title,
                        'doc_id': doc_id,
                        'embedding': None
                    })
        
        return concepts
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common to be a useful concept."""
        common_words = {
            'paper', 'study', 'research', 'work', 'method', 'approach',
            'result', 'analysis', 'discussion', 'conclusion', 'introduction',
            'figure', 'table', 'section', 'chapter', 'example', 'case'
        }
        return word in common_words
    
    def _deduplicate_corpus_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate concepts, keeping the best context snippet."""
        concept_groups = defaultdict(list)
        
        # Group by concept name
        for concept in concepts:
            concept_groups[concept['concept']].append(concept)
        
        # Keep best example of each concept
        deduplicated = []
        for concept_name, group in concept_groups.items():
            # Sort by snippet length and informativeness
            best_concept = max(group, key=lambda x: len(x['snippet']))
            deduplicated.append(best_concept)
        
        return deduplicated
    
    def sample_pairs(self, paper_concepts: List[Dict[str, Any]], 
                    corpus_concepts: List[Dict[str, Any]],
                    candidate_pool_size: int = 15000,
                    target_pairs: int = 1000) -> List[Dict[str, Any]]:
        """
        Sample paper×corpus concept pairs using importance sampling.
        
        Args:
            paper_concepts: Concepts extracted from the paper
            corpus_concepts: Concepts from the corpus
            candidate_pool_size: Size of initial candidate pool
            target_pairs: Number of pairs to return
            
        Returns:
            List of sampled pairs with metadata
        """
        # Compute embeddings for corpus concepts if needed
        corpus_concepts = self._compute_corpus_embeddings(corpus_concepts)
        
        # Generate candidate pairs
        candidates = self._generate_candidate_pairs(
            paper_concepts, corpus_concepts, candidate_pool_size
        )
        
        # Score pairs for importance sampling
        scored_pairs = self._score_pairs(candidates)
        
        # Sample pairs using importance sampling
        sampled_pairs = self._importance_sample(scored_pairs, target_pairs)
        
        return sampled_pairs
    
    def _compute_corpus_embeddings(self, corpus_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute embeddings for corpus concepts that don't have them."""
        for concept in corpus_concepts:
            if concept['embedding'] is None:
                text_to_embed = concept['concept'] + " " + concept['snippet']
                concept['embedding'] = self.embedding_model.encode(text_to_embed).tolist()
        
        return corpus_concepts
    
    def _generate_candidate_pairs(self, paper_concepts: List[Dict[str, Any]], 
                                 corpus_concepts: List[Dict[str, Any]],
                                 pool_size: int) -> List[Dict[str, Any]]:
        """Generate candidate paper×corpus pairs."""
        candidates = []
        
        # Simple combinatorial approach - later we can add smarter pre-filtering
        for paper_concept in paper_concepts:
            for corpus_concept in corpus_concepts:
                # Skip if same concept name (too obvious)
                if paper_concept['concept'].lower() == corpus_concept['concept'].lower():
                    continue
                
                candidates.append({
                    'paper_concept': paper_concept,
                    'corpus_concept': corpus_concept,
                    'paper_name': paper_concept['concept'],
                    'corpus_name': corpus_concept['concept'],
                    'author': corpus_concept['author'],
                    'year': corpus_concept['year'],
                    'title': corpus_concept['title']
                })
        
        # Randomly shuffle and take subset if too many
        self.random.shuffle(candidates)
        return candidates[:pool_size]
    
    def _score_pairs(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score pairs for novelty and plausibility."""
        for pair in candidates:
            paper_emb = np.array(pair['paper_concept']['embedding']).reshape(1, -1)
            corpus_emb = np.array(pair['corpus_concept']['embedding']).reshape(1, -1)
            
            # Novelty score (1 - cosine similarity)
            similarity = cosine_similarity(paper_emb, corpus_emb)[0][0]
            novelty = 1 - similarity
            
            # Plausibility score (small lexical overlap or shared semantic space)
            plausibility = self._compute_plausibility(pair)
            
            # Diversity score (author, year, topic variety)
            diversity = self._compute_diversity_score(pair)
            
            # Combined score
            pair['novelty'] = novelty
            pair['plausibility'] = plausibility
            pair['diversity'] = diversity
            pair['score'] = novelty * 0.5 + plausibility * 0.3 + diversity * 0.2
        
        return candidates
    
    def _compute_plausibility(self, pair: Dict[str, Any]) -> float:
        """Compute plausibility score for a concept pair."""
        paper_concept = pair['paper_concept']['concept'].lower()
        corpus_concept = pair['corpus_concept']['concept'].lower()
        
        # Check for shared words (but not too many)
        paper_words = set(paper_concept.split())
        corpus_words = set(corpus_concept.split())
        shared_words = paper_words & corpus_words
        
        # Some overlap is good, too much is too obvious
        if len(shared_words) == 1:
            return 0.8  # One shared word suggests connection
        elif len(shared_words) == 0:
            # Check for semantic relatedness through common technical domains
            tech_domains = [
                ['neural', 'network', 'deep', 'learning'],
                ['attention', 'transformer', 'embedding'],
                ['reinforcement', 'policy', 'reward'],
                ['sparse', 'autoencoder', 'representation'],
                ['interpretability', 'explanation', 'visualization']
            ]
            
            for domain in tech_domains:
                paper_in_domain = any(word in paper_concept for word in domain)
                corpus_in_domain = any(word in corpus_concept for word in domain)
                if paper_in_domain and corpus_in_domain:
                    return 0.6
            
            return 0.4  # Completely unrelated might still be interesting
        else:
            return 0.2  # Too much overlap is too obvious
    
    def _compute_diversity_score(self, pair: Dict[str, Any]) -> float:
        """Compute diversity score to encourage variety in sampling."""
        author = pair['author']
        year = pair['year']
        
        # Higher score for unseen authors and years
        author_penalty = 0.5 if author in self.used_authors else 1.0
        year_penalty = 0.7 if year in self.used_years else 1.0
        
        return author_penalty * year_penalty
    
    def _importance_sample(self, scored_pairs: List[Dict[str, Any]], 
                          target_count: int) -> List[Dict[str, Any]]:
        """Sample pairs using importance sampling based on scores."""
        # Sort by score
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # Use reservoir sampling with importance weights for diversity
        sampled = []
        
        # Take top candidates with some randomness
        top_candidates = scored_pairs[:min(target_count * 3, len(scored_pairs))]
        
        # Weighted sampling from top candidates
        weights = np.array([pair['score'] for pair in top_candidates])
        
        # Ensure weights are positive and normalize
        weights = np.maximum(weights, 1e-10)  # Avoid zero weights
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        sampled_indices = np.random.choice(
            len(top_candidates), 
            size=min(target_count, len(top_candidates)),
            replace=False,
            p=weights
        )
        
        for idx in sampled_indices:
            pair = top_candidates[idx]
            sampled.append(pair)
            
            # Update diversity tracking
            self.used_authors.add(pair['author'])
            self.used_years.add(pair['year'])
            self.sampled_pairs.add((pair['paper_name'], pair['corpus_name']))
        
        return sampled
    
    def save_pairs(self, pairs: List[Dict[str, Any]], output_path: str):
        """Save sampled pairs to JSONL file."""
        with open(output_path, 'w') as f:
            for pair in pairs:
                # Create serializable version for the generator
                serializable_pair = {
                    'paper_concept': pair['paper_name'],
                    'corpus_concept': pair['corpus_name'],
                    'paper_backpack': pair['paper_concept']['backpack'],
                    'paper_section_title': pair['paper_concept'].get('section_title'),
                    'corpus_snippet': pair['corpus_concept']['snippet'],
                    'author': pair['author'],
                    'year': pair['year'],
                    'title': pair['title'],
                    'doc_id': pair['corpus_concept'].get('doc_id', 'unknown'),
                    'novelty': pair['novelty'],
                    'plausibility': pair['plausibility'],
                    'diversity': pair['diversity'],
                    'score': pair['score'],
                    # Include anchor information for better binding compatibility
                    'paper_anchor_exact': pair['paper_concept'].get('anchor_exact', pair['paper_name']),
                    'paper_anchor_alias': pair['paper_concept'].get('anchor_alias', pair['paper_name'])
                }
                f.write(json.dumps(serializable_pair) + '\n')


if __name__ == "__main__":
    # Test the sampler
    sampler = EphemeralDDLSampler()
    
    # Mock paper concepts
    paper_concepts = [
        {
            'concept': 'sparse autoencoder',
            'backpack': 'Sparse autoencoders learn compressed representations...',
            'embedding': [0.1, 0.2, 0.3]  # Mock embedding
        },
        {
            'concept': 'steering vector',
            'backpack': 'Steering vectors control model behavior...',
            'embedding': [0.2, 0.3, 0.1]
        }
    ]
    
    # Mock corpus concepts
    corpus_concepts = [
        {
            'concept': 'attention mechanism',
            'snippet': 'The attention mechanism allows models to focus...',
            'author': 'Smith',
            'year': 2023,
            'title': 'Attention Research',
            'doc_id': 'test1',
            'embedding': [0.3, 0.1, 0.2]
        }
    ]
    
    pairs = sampler.sample_pairs(paper_concepts, corpus_concepts, 100, 10)
    print(f"Sampled {len(pairs)} pairs")
    for pair in pairs[:3]:
        print(f"- {pair['paper_name']} × {pair['corpus_name']} (score: {pair['score']:.3f})")