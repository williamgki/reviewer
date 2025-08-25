#!/usr/bin/env python3

import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

class SemanticDDLPipeline:
    """
    Complete DDL pipeline using:
    1. LLM for paper concept extraction (already implemented)
    2. Semantic search for corpus concept discovery  
    3. Intelligent pairing for meaningful connections
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 🔧 CONFIG DIAGNOSTIC: Log config receipt in semantic pipeline
        retrieval_config = config.get('retrieval', {})
        llm_config = config.get('llm', {})
        print(f"🔧 SemanticDDL CONFIG: min_semantic_score={retrieval_config.get('min_semantic_score', 'NOT_SET')}, "
              f"top_k={retrieval_config.get('top_k', 'NOT_SET')}, "
              f"model={llm_config.get('model', config.get('model', 'NOT_SET'))}")
        
        # Import existing components
        from paper_concepts import PaperConceptExtractor
        from semantic_corpus_sampler import SemanticCorpusConceptSampler
        
        # Initialize paper concept extractor (LLM-based)
        self.paper_extractor = PaperConceptExtractor(
            use_llm_extraction=True,
            api_base=config.get('llm', {}).get('api_base', config.get('api_base', 'http://localhost:1234/v1')),
            api_key=config.get('api_key', 'lm-studio'),
            model=config.get('llm', {}).get('model', config.get('model', 'gpt-oss-120b'))
        )
        
        # Initialize semantic corpus sampler with config
        self.corpus_sampler = SemanticCorpusConceptSampler(config=config)
    
    def generate_concept_pairs(self, paper_text: str, target_pairs: int = 10) -> List[Dict[str, Any]]:
        """
        Generate high-quality paper × corpus concept pairs using semantic matching.
        """
        print(f"🎯 Generating {target_pairs} high-quality concept pairs...")
        
        # Phase 1: Extract paper concepts using LLM
        print("📄 Phase 1: Extracting paper concepts (LLM-based)...")
        paper_concepts = self.paper_extractor.extract_concepts(
            paper_text, 
            max_concepts=min(target_pairs, 15)  # Don't need too many paper concepts
        )
        
        if not paper_concepts:
            print("❌ No paper concepts extracted")
            return []
        
        # Filter out low-quality concepts  
        filtered_concepts = self._filter_quality_concepts(paper_concepts)
        
        print(f"   ✅ Extracted {len(filtered_concepts)} paper concepts (filtered from {len(paper_concepts)}):")
        for concept in filtered_concepts[:5]:  # Show first 5
            print(f"      • {concept['concept']}")
        
        paper_concepts = filtered_concepts
        
        # Phase 2: Find semantically relevant corpus concepts
        print("🔍 Phase 2: Finding relevant corpus concepts (semantic search)...")
        
        semantic_pairs = []
        concepts_per_paper = max(1, target_pairs // len(paper_concepts))
        
        for paper_concept in paper_concepts:
            if len(semantic_pairs) >= target_pairs:
                break
                
            # Use semantic search to find related corpus chunks
            relevant_corpus_concepts = self.corpus_sampler.find_relevant_corpus_concepts(
                [paper_concept], 
                concepts_per_paper_concept=concepts_per_paper * 2  # Get extras for quality filtering
            )
            
            # Phase 3: Create intelligent pairs
            for corpus_concept in relevant_corpus_concepts[:concepts_per_paper]:
                if len(semantic_pairs) >= target_pairs:
                    break
                
                pair = self._create_concept_pair(paper_concept, corpus_concept)
                semantic_pairs.append(pair)
        
        print(f"   ✅ Generated {len(semantic_pairs)} semantic concept pairs")
        
        # Phase 3: Deduplicate pairs to avoid "AI alignment" + "ai alignment"
        print("🔧 Phase 3: Deduplicating concept pairs...")
        deduped_pairs = self._deduplicate_pairs(semantic_pairs)
        print(f"   Removed {len(semantic_pairs) - len(deduped_pairs)} duplicate pairs")
        
        # Phase 4: Quality scoring and ranking
        print("⚡ Phase 4: Scoring and ranking pairs...")
        scored_pairs = self._score_and_rank_pairs(deduped_pairs)
        
        final_pairs = scored_pairs[:target_pairs]
        
        print(f"🎉 Final result: {len(final_pairs)} high-quality pairs")
        self._print_pair_sample(final_pairs)
        
        # Print metrics one-liner
        unique_sources = len(set(pair.get('doc_id', 'unknown') for pair in final_pairs))
        avg_score = sum(pair.get('combined_score', 0) for pair in final_pairs) / len(final_pairs) if final_pairs else 0
        
        print(f"📊 METRICS: concepts={len(paper_concepts)} pairs={len(final_pairs)} sources={unique_sources} avg_score={avg_score:.3f}")
        
        # Auto-relaxation tripwire
        if unique_sources < 3 and len(final_pairs) > 0:
            print(f"⚠️ TRIPWIRE: Only {unique_sources} unique sources, may need relaxation for next run")
        
        # Validate schema before returning
        self._validate_pair_schema(final_pairs)
        
        return final_pairs
    
    def _validate_pair_schema(self, pairs: List[Dict[str, Any]]) -> None:
        """Validate pair schema to prevent downstream crashes."""
        if not pairs:
            return
            
        required_fields = {
            'paper_concept': dict,  # Must be dict for sampler
            'paper_name': str,
            'corpus_concept': dict,  # Must be dict for sampler (has snippet, doc_id)
            'corpus_name': str,
            'doc_id': str
        }
        
        for i, pair in enumerate(pairs[:3]):  # Check first few pairs
            for field, expected_type in required_fields.items():
                if field not in pair:
                    raise ValueError(f"Pair {i}: Missing required field '{field}'")
                if not isinstance(pair[field], expected_type):
                    actual_type = type(pair[field]).__name__
                    raise ValueError(f"Pair {i}: Field '{field}' must be {expected_type.__name__}, got {actual_type}")
        
        print(f"✅ Schema validation passed for {len(pairs)} pairs")
    
    def _filter_quality_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced concept filtering with domain-specific improvements."""
        filtered = []
        
        for concept in concepts:
            concept_name = concept.get('concept', '')
            
            # Skip all-caps site names (like "LESSWRONG", "ARXIV")
            if concept_name.isupper() and len(concept_name) > 2:
                print(f"    ❌ Filtering all-caps site: '{concept_name}'")
                continue
                
            # Skip very short concepts or single letters
            if len(concept_name.strip()) < 3:
                print(f"    ❌ Filtering too short: '{concept_name}'")
                continue
                
            # Skip common stop-like concepts and generic terms
            stop_concepts = {
                'the', 'and', 'with', 'this', 'that', 'from', 'they', 'them', 'their',
                'important', 'interesting', 'significant', 'relevant', 'useful', 'key',
                'approach', 'method', 'system', 'model', 'paper', 'study', 'research'
            }
            if concept_name.lower().strip() in stop_concepts:
                print(f"    ❌ Filtering stop concept: '{concept_name}'")
                continue
            
            # Skip pure numbers or alphanumeric codes
            if concept_name.isdigit() or (len(concept_name) < 6 and any(c.isdigit() for c in concept_name)):
                print(f"    ❌ Filtering numeric/code: '{concept_name}'")
                continue
            
            # Enhance with domain synonyms if this is a key concept
            enhanced_concept = self._enhance_with_domain_knowledge(concept)
            filtered.append(enhanced_concept)
        
        return filtered
    
    def _enhance_with_domain_knowledge(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance concept with domain-specific synonyms and context."""
        concept_name = concept.get('concept', '').lower()
        
        # Add domain context for key AI safety terms
        domain_enhancements = {
            'truthfulness': 'AI truthfulness and factuality in language models',
            'alignment': 'AI alignment and value alignment for safe AI systems',
            'honesty': 'AI honesty policy and truthful AI behavior',
            'deception': 'AI deception detection and prevention in language models',
            'calibration': 'model calibration and confidence estimation in AI systems',
            'hallucination': 'AI hallucination and factual accuracy in language models',
            'sycophancy': 'AI sycophancy and people-pleasing behavior in language models'
        }
        
        # Check if concept contains key terms
        for key_term, enhancement in domain_enhancements.items():
            if key_term in concept_name:
                # Add enhanced backpack context
                original_backpack = concept.get('backpack', '')
                if not original_backpack or len(original_backpack) < 50:
                    concept['backpack'] = f"{original_backpack} {enhancement}".strip()
                break
        
        return concept
    
    def _deduplicate_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate pairs based on case-insensitive paper concept + doc_id."""
        seen = set()
        deduped = []
        
        # Sort by score first to keep highest-scoring duplicates
        for pair in sorted(pairs, key=lambda x: x.get('score', 0.0), reverse=True):
            # Handle paper_concept as dict (get concept name)
            paper_concept = pair.get('paper_concept', '')
            if isinstance(paper_concept, dict):
                paper_key = paper_concept.get('concept', '').casefold()
            else:
                paper_key = str(paper_concept).casefold()
            doc_key = pair.get('doc_id', 'unknown')
            key = (paper_key, doc_key)
            
            if key not in seen:
                seen.add(key)
                deduped.append(pair)
        
        return deduped
    
    def _create_concept_pair(self, paper_concept: Dict[str, Any], 
                           corpus_concept: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured concept pair for DDL generation with complete schema."""
        
        # Extract metadata (corpus_concept comes from search results)
        meta = corpus_concept.get('metadata', {}) or {}
        snippet = corpus_concept.get('snippet', '')[:200]
        
        return {
            # Paper side - preserve full structure for sampler compatibility
            'paper_concept': paper_concept,  # Keep as dict - sampler expects this structure
            'paper_name': paper_concept['concept'],  # Compatibility with sampler
            'paper_anchor_exact': paper_concept.get('anchor_exact', paper_concept['concept']),
            'paper_anchor_alias': paper_concept.get('anchor_alias', paper_concept['concept']),
            'paper_backpack': paper_concept.get('backpack', ''),
            
            # Corpus side - complete metadata for binder/critic stages 
            'corpus_concept': {
                'concept': corpus_concept.get('concept') or snippet[:64] or 'Unknown',
                'snippet': snippet,
                'doc_id': corpus_concept.get('doc_id', 'unknown'),
                'source': 'semantic_search'
            },
            'corpus_name': corpus_concept.get('concept') or snippet[:64] or 'Unknown',  # Compatibility
            'doc_id': corpus_concept.get('doc_id', 'unknown'),
            'source': meta.get('source') or meta.get('title') or corpus_concept.get('title', 'Unknown'),
            'title': corpus_concept.get('title') or meta.get('title', 'Unknown'),
            'authors': meta.get('authors', []) or [corpus_concept.get('author', 'Unknown')],
            'author': corpus_concept.get('author') or meta.get('author', 'Unknown'),  # For backward compatibility
            'year': corpus_concept.get('year') or meta.get('year', 2020),
            'url': meta.get('url', ''),
            
            # Evidence structure for binder - windowed snippets
            'evidence': {
                'snippet': self._create_windowed_snippet(snippet, paper_concept['concept']),
                'start': meta.get('start_char'),
                'end': meta.get('end_char'),
                'anchor': meta.get('anchor') or paper_concept['concept'],
                'source': 'corpus',
                'content': corpus_concept.get('snippet', '')[:500],  # Longer content for context
                'doc_id': corpus_concept.get('doc_id', 'unknown'),  # Ensure doc_id for distinct doc requirement
                'evidence_quality': 'windowed'  # Mark as high-quality evidence
            },
            
            # Relevance scores
            'semantic_score': corpus_concept.get('dense_score', 0.0),
            'lexical_score': corpus_concept.get('bm25_score', 0.0),
            'combined_score': corpus_concept.get('combined_score', 0.0),
            'score': corpus_concept.get('combined_score', 0.0),  # Main score for sorting
            'paper_concept_importance': paper_concept.get('importance', 5.0),
            
            # Quality metrics
            'novelty': self._calculate_novelty(paper_concept, corpus_concept),
            'plausibility': self._calculate_plausibility(paper_concept, corpus_concept),
            'diversity': 1.0,  # Will be calculated relative to other pairs
        }
    
    def _calculate_novelty(self, paper_concept: Dict[str, Any], 
                         corpus_concept: Dict[str, Any]) -> float:
        """Calculate novelty score for the concept pair."""
        
        # Higher novelty for different domains/contexts
        paper_terms = set(paper_concept['concept'].lower().split())
        corpus_terms = set(corpus_concept['concept'].lower().split())
        
        # Jaccard distance as novelty measure
        intersection = len(paper_terms & corpus_terms)
        union = len(paper_terms | corpus_terms)
        
        if union == 0:
            return 0.5
        
        jaccard_similarity = intersection / union
        novelty = 1.0 - jaccard_similarity  # More different = more novel
        
        # Bonus for cross-domain connections
        if corpus_concept.get('combined_score', 0) > 0.5:  # High semantic similarity
            novelty += 0.2  # Bonus for meaningful but different concepts
        
        return min(novelty, 1.0)
    
    def _calculate_plausibility(self, paper_concept: Dict[str, Any], 
                              corpus_concept: Dict[str, Any]) -> float:
        """Calculate plausibility score for the concept pair."""
        
        # Base plausibility from semantic similarity
        semantic_score = corpus_concept.get('combined_score', 0.0)
        plausibility = semantic_score * 0.8  # Scale to 0-0.8 range
        
        # Bonus for high-importance paper concepts
        paper_importance = paper_concept.get('importance', 5.0)
        if paper_importance > 7.0:
            plausibility += 0.2
        
        # Penalty for very short or generic concepts
        if (len(corpus_concept['concept']) < 5 or 
            corpus_concept['concept'].lower() in {'the', 'and', 'with', 'about', 'from'}):
            plausibility *= 0.5
        
        return min(plausibility, 1.0)
    
    def _create_windowed_snippet(self, snippet: str, anchor: str) -> str:
        """Create focused 1-2 sentence window around anchor for quality evidence."""
        import re
        
        if not snippet or not anchor:
            return snippet[:200] if snippet else ''
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', snippet)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return snippet[:200]
        
        # Find sentence containing anchor (case insensitive)
        anchor_lower = anchor.lower()
        anchor_sentence_idx = -1
        
        for i, sentence in enumerate(sentences):
            if anchor_lower in sentence.lower():
                anchor_sentence_idx = i
                break
        
        # If anchor not found, take first 1-2 sentences
        if anchor_sentence_idx == -1:
            window_sentences = sentences[:2]
        else:
            # Take sentence with anchor + 1 before/after for context
            start_idx = max(0, anchor_sentence_idx - 1)
            end_idx = min(len(sentences), anchor_sentence_idx + 2)
            window_sentences = sentences[start_idx:end_idx]
        
        windowed = '. '.join(window_sentences)
        
        # Cap at reasonable length
        if len(windowed) > 300:
            windowed = windowed[:300] + '...'
        
        return windowed
    
    def _score_and_rank_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank pairs by overall quality."""
        
        for pair in pairs:
            # Combined quality score
            quality_score = (
                pair['novelty'] * 0.3 +
                pair['plausibility'] * 0.4 + 
                pair['combined_score'] * 0.2 +
                (pair['paper_concept_importance'] / 10.0) * 0.1
            )
            
            pair['score'] = quality_score
        
        # Sort by quality score
        pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate diversity bonus (penalize very similar pairs)
        self._apply_diversity_bonus(pairs)
        
        return pairs
    
    def _apply_diversity_bonus(self, pairs: List[Dict[str, Any]]):
        """Apply diversity bonus to encourage variety in concept pairs."""
        
        seen_paper_concepts = set()
        seen_corpus_domains = set()
        
        for i, pair in enumerate(pairs):
            # Handle paper_concept as dict
            paper_concept_obj = pair.get('paper_concept', '')
            if isinstance(paper_concept_obj, dict):
                paper_concept = paper_concept_obj.get('concept', '').lower()
            else:
                paper_concept = str(paper_concept_obj).lower()
            corpus_title = pair.get('title', '').lower()
            
            # Diversity penalty for repeated paper concepts
            if paper_concept in seen_paper_concepts:
                pair['score'] *= 0.8
            else:
                seen_paper_concepts.add(paper_concept)
            
            # Diversity penalty for same corpus domain
            corpus_domain = corpus_title.split()[:3]  # First 3 words as domain indicator
            corpus_domain_key = ' '.join(corpus_domain)
            
            if corpus_domain_key in seen_corpus_domains:
                pair['score'] *= 0.9
            else:
                seen_corpus_domains.add(corpus_domain_key)
        
        # Re-sort after diversity adjustment
        pairs.sort(key=lambda x: x['score'], reverse=True)
    
    def _print_pair_sample(self, pairs: List[Dict[str, Any]]):
        """Print sample of generated pairs for inspection."""
        
        print("\n📋 Sample of generated pairs:")
        for i, pair in enumerate(pairs[:3]):  # Show top 3 pairs
            print(f"   {i+1}. '{pair['paper_concept']}' + '{pair['corpus_concept']}'")
            print(f"      Quality: {pair['score']:.3f} | Semantic: {pair['combined_score']:.3f}")
            print(f"      Source: {pair['title'][:50]}...")
            print()

# Integration with existing DDL pipeline
def create_enhanced_ddl_pipeline(config: Dict[str, Any]):
    """Create enhanced DDL pipeline with semantic concept pairing."""
    
    semantic_pipeline = SemanticDDLPipeline(config)
    
    return semantic_pipeline

# Test function
def test_semantic_ddl_pipeline():
    """Test the complete semantic DDL pipeline."""
    
    config = {
        'api_base': 'http://localhost:1234/v1',
        'api_key': 'lm-studio',
        'model': 'gpt-oss-120b'
    }
    
    # Test with honest.txt
    paper_path = "pipeline/input/honest.txt"
    
    if not os.path.exists(paper_path):
        print(f"❌ Test paper not found: {paper_path}")
        return
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    
    print("🧪 Testing Semantic DDL Pipeline...")
    print(f"   Paper: {len(paper_text)} characters")
    
    pipeline = create_enhanced_ddl_pipeline(config)
    pairs = pipeline.generate_concept_pairs(paper_text, target_pairs=5)
    
    print(f"\n✅ Generated {len(pairs)} pairs for testing")
    
    if pairs:
        print("\n🎯 Quality Analysis:")
        avg_semantic = np.mean([p['combined_score'] for p in pairs])
        avg_quality = np.mean([p['score'] for p in pairs])
        
        print(f"   Average semantic score: {avg_semantic:.3f}")
        print(f"   Average quality score: {avg_quality:.3f}")
        
        domain_relevant = sum(1 for p in pairs 
                            if any(term in p['corpus_concept'].lower() for term in [
                                'learning', 'ai', 'model', 'alignment', 'safety', 'neural'
                            ]))
        
        print(f"   Domain-relevant pairs: {domain_relevant}/{len(pairs)} ({100*domain_relevant/len(pairs):.1f}%)")

if __name__ == "__main__":
    test_semantic_ddl_pipeline()