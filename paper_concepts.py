#!/usr/bin/env python3

import json
import re
import spacy
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Import LLM-based extractor
try:
    from llm_concept_extractor import LLMConceptExtractor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM concept extractor not available - using fallback methods")


class PaperConceptExtractor:
    """
    Stateless extraction of core concepts from a research paper.
    Extracts noun phrases, method names, mechanisms, entities with context backpacks.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", 
                 use_llm_extraction: bool = True,
                 api_base: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio",
                 model: str = "gpt-oss-120b"):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("WARNING: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize LLM-based extraction if requested
        self.use_llm_extraction = use_llm_extraction and LLM_AVAILABLE
        if self.use_llm_extraction:
            self.llm_extractor = LLMConceptExtractor(api_base=api_base, api_key=api_key, model=model)
        else:
            self.llm_extractor = None
        
        # Stop words to filter out
        self.stop_words = {
            'paper', 'study', 'research', 'work', 'approach', 'method', 'technique',
            'analysis', 'results', 'conclusion', 'discussion', 'introduction',
            'related', 'previous', 'prior', 'existing', 'proposed', 'novel',
            'new', 'different', 'various', 'several', 'many', 'most', 'some',
            'important', 'significant', 'main', 'key', 'major', 'primary'
        }
        
        # Domain-specific technical terms that should be preserved
        self.preserve_terms = {
            'neural network', 'machine learning', 'deep learning', 'transformer',
            'attention mechanism', 'gradient descent', 'backpropagation',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'sparse autoencoder', 'steering vector', 'activation patching',
            'mechanistic interpretability', 'feature visualization', 'concept bottleneck'
        }

    def extract_concepts(self, paper_text: str, max_concepts: int = 30) -> List[Dict[str, Any]]:
        """
        Extract core concepts from paper text with context backpacks.
        
        Args:
            paper_text: Full text of the research paper
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of concept dictionaries with name, source, backpack, and embedding
        """
        if self.use_llm_extraction and self.llm_extractor:
            print("Using LLM-based semantic concept extraction for better domain relevance")
            return self._extract_concepts_llm(paper_text, max_concepts)
        else:
            # FORCE enhanced extraction method - better results than spaCy-based approach
            print("Using enhanced n-gram concept extraction (forced for better AI safety domain coverage)")
            return self._extract_concepts_fallback(paper_text, max_concepts)
    
    def _extract_candidate_concepts(self, doc) -> List[Dict[str, Any]]:
        """Extract candidate concepts using spaCy NLP."""
        candidates = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            text = chunk.text.lower().strip()
            if self._is_valid_concept(text):
                candidates.append({
                    'name': text,
                    'type': 'noun_phrase',
                    'freq': 1,
                    'span': (chunk.start_char, chunk.end_char)
                })
        
        # Extract named entities (technical terms, methods)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'FAC', 'WORK_OF_ART']:
                text = ent.text.lower().strip()
                if self._is_valid_concept(text):
                    candidates.append({
                        'name': text,
                        'type': 'named_entity',
                        'freq': 1,
                        'span': (ent.start_char, ent.end_char)
                    })
        
        # Extract compound technical terms (pattern matching)
        text_lower = doc.text.lower()
        for term in self.preserve_terms:
            if term in text_lower:
                candidates.append({
                    'name': term,
                    'type': 'technical_term',
                    'freq': text_lower.count(term),
                    'span': None
                })
        
        return candidates
    
    def _is_valid_concept(self, text: str) -> bool:
        """Check if a concept candidate is valid."""
        # Minimum length
        if len(text) < 4:
            return False
            
        # No stop words
        if text in self.stop_words:
            return False
            
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False
            
        # No single common words
        if text in ['data', 'model', 'system', 'algorithm', 'function', 'value', 'result']:
            return False
            
        return True
    
    def _filter_and_score_concepts(self, candidates: List[Dict], paper_text: str) -> List[Dict]:
        """Filter candidates and assign relevance scores."""
        # Count frequency of each concept
        concept_counts = Counter()
        concept_data = {}
        
        for candidate in candidates:
            name = candidate['name']
            concept_counts[name] += candidate['freq']
            if name not in concept_data:
                concept_data[name] = candidate
        
        # Score concepts
        scored_concepts = []
        text_lower = paper_text.lower()
        
        for concept, freq in concept_counts.items():
            # Base score from frequency (log to reduce impact of very common terms)
            freq_score = min(np.log(freq + 1) / 5.0, 1.0)
            
            # Boost for technical terms
            tech_boost = 1.5 if concept in self.preserve_terms else 1.0
            
            # Boost for concepts that appear in different contexts
            context_variety = len(set(re.findall(rf'.{{0,50}}{re.escape(concept)}.{{0,50}}', text_lower)))
            variety_score = min(context_variety / 3.0, 1.0)
            
            # Penalize very short or very common concepts
            length_penalty = 0.5 if len(concept) < 6 else 1.0
            
            final_score = freq_score * tech_boost * variety_score * length_penalty
            
            scored_concepts.append({
                'name': concept,
                'score': final_score,
                'freq': freq,
                'type': concept_data[concept]['type']
            })
        
        return scored_concepts
    
    def _generate_context_backpack(self, concept: str, paper_text: str, 
                                 target_tokens: int = 150) -> str:
        """Generate a context backpack snippet for a concept."""
        sentences = re.split(r'[.!?]+', paper_text)
        relevant_sentences = []
        
        # Find sentences containing the concept
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                # Clean and trim sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:  # Ignore very short fragments
                    relevant_sentences.append(clean_sentence)
        
        if not relevant_sentences:
            return ""
        
        # Select best sentences (prefer ones with more context)
        scored_sentences = []
        for sent in relevant_sentences[:10]:  # Limit to avoid processing too many
            # Score based on length and information density
            word_count = len(sent.split())
            info_density = len([w for w in sent.split() if len(w) > 4]) / max(word_count, 1)
            score = min(word_count / 20.0, 1.0) * info_density
            scored_sentences.append((sent, score))
        
        # Take top sentences up to token limit
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        backpack = ""
        for sent, _ in scored_sentences:
            if len((backpack + " " + sent).split()) <= target_tokens:
                backpack = (backpack + " " + sent).strip()
            else:
                break

        return backpack

    def _generate_multi_scale_backpacks(self, concept: str, paper_text: str) -> Dict[str, Any]:
        """Generate small, medium, and large context backpacks plus section title."""
        return {
            'backpack_s': self._generate_context_backpack(concept, paper_text, target_tokens=50),
            'backpack_m': self._generate_context_backpack(concept, paper_text, target_tokens=150),
            'backpack_l': self._generate_context_backpack(concept, paper_text, target_tokens=300),
            'section_title': None,
        }

    def _extract_concepts_fallback(self, paper_text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """Enhanced n-gram concept extraction with domain awareness and anchor bundling."""
        print("Using enhanced n-gram concept extraction (fallback mode)")
        
        from collections import Counter
        import math
        
        # Enhanced stopwords - more comprehensive
        BASIC_STOP = {
            "the", "a", "an", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
            "of", "for", "to", "in", "on", "by", "as", "and", "or", "but", "if", "then", "so", "thus", 
            "there", "here", "it", "they", "them", "we", "you", "i", "he", "she", "my", "your", "their",
            "very", "really", "just", "also", "much", "many", "more", "most", "less", "least", "may", "might"
        }
        
        # Domain junk - filter out as unigrams only
        DOMAIN_STOP_UNIGRAMS = {"ai", "model", "models", "system", "paper", "study", "work", "approach", "method"}
        
        # Domain boost - known AI safety terms get priority
        DOMAIN_BOOST = {
            "mechanistic interpretability": 2.2, "feature attribution": 1.5, "activation patching": 1.8,
            "sparse autoencoder": 2.0, "dictionary learning": 1.5, "steering vector": 1.8,
            "deceptive alignment": 2.2, "mesa optimizer": 2.0, "inner alignment": 1.8,
            "capability evaluation": 1.6, "red teaming": 1.4, "reward modeling": 1.6, "rlhf": 1.6,
            "specification gaming": 1.8, "goal misgeneralization": 1.8, "distributional shift": 1.5,
            "chain of thought": 1.4, "few shot": 1.3, "zero shot": 1.3, "in context learning": 1.5,
            "transformer architecture": 1.4, "attention mechanism": 1.4, "residual stream": 1.6,
            "gradient descent": 1.3, "backpropagation": 1.3, "neural network": 1.2,
            "machine learning": 1.2, "deep learning": 1.2, "reinforcement learning": 1.3,
            "supervised learning": 1.3, "unsupervised learning": 1.3, "representation learning": 1.4,
            "concept bottleneck": 1.6, "feature visualization": 1.5, "saliency map": 1.4,
            "adversarial example": 1.5, "robustness": 1.3, "generalization": 1.3,
            "safety case": 1.8, "ai safety": 1.6, "alignment tax": 1.8, "capability control": 1.7
        }
        
        # Section weights - boost title and abstract more
        SEC_WEIGHTS = {"title": 8.0, "abstract": 4.0, "headings": 2.5, "body": 1.0}
        
        def normalize(s: str) -> str:
            """Normalize text for consistent processing."""
            s = s.lower()
            s = re.sub(r"[''`]", "'", s)  # Normalize quotes
            s = re.sub(r"[\u2013\u2014-]", "-", s)  # Normalize dashes
            s = re.sub(r"\b(a\.?i\.?s?)\b", "ai", s)  # Normalize AI variants
            s = re.sub(r"[^a-z0-9\- ]+", " ", s)  # Keep only alphanumeric, hyphens, spaces
            s = re.sub(r"\s+", " ", s).strip()  # Collapse whitespace
            return s
        
        def is_bad_unigram(tok: str) -> bool:
            """Check if a unigram should be filtered out."""
            return (tok in BASIC_STOP) or (tok in DOMAIN_STOP_UNIGRAMS) or len(tok) < 3
        
        def get_head_noun(phrase: str) -> str:
            """Extract head noun (last meaningful word) from phrase."""
            tokens = phrase.split()
            # Return last non-stopword token, or last token if all are stopwords
            for tok in reversed(tokens):
                if tok not in BASIC_STOP:
                    return tok
            return tokens[-1] if tokens else phrase
        
        def extract_candidates(section_text: str):
            """Extract candidate n-gram phrases with improved filtering."""
            s = normalize(section_text)
            tokens = s.split()
            
            # Remove basic stopwords from tokens for n-gram generation
            filtered_tokens = []
            for t in tokens:
                if len(t) > 2 and t not in BASIC_STOP:
                    filtered_tokens.append(t)
            
            phrases = []
            # Generate n-grams 2-5 (prioritize multi-word phrases)
            for n in range(2, 6):
                for i in range(len(filtered_tokens) - n + 1):
                    ng = filtered_tokens[i:i+n]
                    phrase = " ".join(ng)
                    
                    # Skip if phrase is too short or starts with number
                    if len(phrase) < 4 or re.match(r"^\d", phrase):
                        continue
                        
                    phrases.append(phrase)
            
            # Also include some carefully selected unigrams (non-stopwords, domain-relevant)
            for token in filtered_tokens:
                if (not is_bad_unigram(token) and 
                    len(token) >= 4 and 
                    not re.match(r"^\d", token)):
                    phrases.append(token)
                    
            return phrases
        
        # 1) Crude but resilient sectioning
        lines = paper_text.splitlines()
        title = lines[0] if lines else ""
        abstract = ""
        headings, body = [], []
        
        for line in lines[1:]:
            line_lower = line.lower().strip()
            if re.match(r"^\s*(abstract|summary)\b[:\-]?\s*", line_lower):
                abstract += " " + line
            elif re.match(r"^\s*#{1,3}\s+|^\s*[A-Z][A-Za-z0-9 \-]{0,60}$", line):
                headings.append(line)
            else:
                body.append(line)
        
        buckets = [
            ("title", title),
            ("abstract", abstract), 
            ("headings", "\n".join(headings)),
            ("body", "\n".join(body))
        ]
        
        # 2) Score candidates by section weight, content quality, and domain relevance
        scores = Counter()
        for sec, sec_text in buckets:
            weight = SEC_WEIGHTS.get(sec, 1.0)
            for phrase in extract_candidates(sec_text):
                content_words = len([w for w in phrase.split() if w not in BASIC_STOP])
                if content_words == 0:
                    continue
                
                # Base score: section weight * content words
                base_score = weight * content_words
                
                # Domain boost for known AI safety terms
                domain_multiplier = DOMAIN_BOOST.get(phrase, 1.0)
                
                # Multi-word bonus (encourage phrases over unigrams)
                word_count = len(phrase.split())
                if word_count >= 2:
                    multiword_bonus = 1.2 + 0.1 * min(word_count - 2, 3)  # 1.2 to 1.5
                else:
                    multiword_bonus = 1.0
                
                final_score = base_score * domain_multiplier * multiword_bonus
                scores[phrase] += final_score
        
        # 3) Deduplicate using head noun + ensure ≥60% multiword phrases
        concepts = []
        seen_heads = set()
        multiword_count = 0
        
        # First pass: prioritize multiword phrases
        for phrase, score in scores.most_common(200):
            word_count = len(phrase.split())
            head = get_head_noun(phrase)
            
            # Skip if head noun already seen
            if head in seen_heads:
                continue
                
            # Generate context backpack and anchor bundle
            backpack = self._generate_context_backpack(phrase, paper_text)
            if not backpack:
                continue
                
            # Create anchor bundle for binding compatibility
            anchor_exact = phrase if phrase in paper_text.lower() else phrase
            anchor_alias = normalize(phrase)
            
            concept_dict = {
                'concept': phrase,
                'source': 'paper', 
                'backpack': backpack,
                'embedding': self.embedding_model.encode(phrase + " " + backpack).tolist(),
                'score': score,
                # Add anchor information for binding compatibility
                'anchor_exact': anchor_exact,
                'anchor_alias': anchor_alias
            }
            
            concepts.append(concept_dict)
            seen_heads.add(head)
            
            if word_count >= 2:
                multiword_count += 1
                
            if len(concepts) >= max_concepts:
                break
        
        # Ensure ≥60% multiword phrases - if not, replace some unigrams
        min_multiword = int(0.6 * max_concepts)
        if multiword_count < min_multiword and len(concepts) < max_concepts:
            # Add more multiword phrases from remaining candidates
            for phrase, score in scores.most_common(400):  # Look deeper
                if len(concepts) >= max_concepts:
                    break
                    
                word_count = len(phrase.split())
                if word_count < 2:  # Only interested in multiword now
                    continue
                    
                head = get_head_noun(phrase)
                if head in seen_heads:
                    continue
                    
                # Skip if already in concepts
                if any(c['concept'] == phrase for c in concepts):
                    continue
                    
                backpack = self._generate_context_backpack(phrase, paper_text)
                if not backpack:
                    continue
                    
                anchor_exact = phrase if phrase in paper_text.lower() else phrase
                anchor_alias = normalize(phrase)
                
                concept_dict = {
                    'concept': phrase,
                    'source': 'paper', 
                    'backpack': backpack,
                    'embedding': self.embedding_model.encode(phrase + " " + backpack).tolist(),
                    'score': score,
                    'anchor_exact': anchor_exact,
                    'anchor_alias': anchor_alias
                }
                
                concepts.append(concept_dict)
                seen_heads.add(head)
                multiword_count += 1
                
                if multiword_count >= min_multiword:
                    break
        
        # Final sort by score and return
        concepts.sort(key=lambda x: x['score'], reverse=True)
        return concepts[:max_concepts]
    
    def save_concepts(self, concepts: List[Dict[str, Any]], output_path: str):
        """Save concepts to JSONL file."""
        with open(output_path, 'w') as f:
            for concept in concepts:
                f.write(json.dumps(concept) + '\n')
    
    def load_concepts(self, input_path: str) -> List[Dict[str, Any]]:
        """Load concepts from JSONL file."""
        concepts = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    concepts.append(json.loads(line))
        return concepts

    def _extract_concepts_llm(self, paper_text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """Extract concepts using LLM semantic understanding."""
        print("Using LLM-based semantic concept extraction")
        
        try:
            # Get concepts from LLM
            llm_concepts = self.llm_extractor.extract_concepts_llm(paper_text, max_concepts)

            # Convert to expected format with embeddings and backpacks
            result_concepts = []

            for concept_data in llm_concepts:
                concept_name = concept_data['name']

                # Create embedding
                try:
                    embedding = self.embedding_model.encode(concept_name).tolist()  # Convert to list for JSON serialization
                except Exception as e:
                    print(f"Warning: Could not create embedding for '{concept_name}': {e}")
                    embedding = np.zeros(384).tolist()  # Default dimension for all-MiniLM-L6-v2
                # Generate multi-scale backpacks from paper context
                backpacks = self._generate_multi_scale_backpacks(concept_name, paper_text)

                llm_backpack = concept_data.get('backpack')
                if llm_backpack:
                    backpacks['backpack_m'] = self._enhance_concept_backpack(
                        concept_name, paper_text, llm_backpack
                    )

                if concept_data.get('section'):
                    backpacks['section_title'] = concept_data['section']

                result_concept = {
                    'concept': concept_name,
                    'source': f"llm_{concept_data.get('section', 'extracted')}",
                    **backpacks,
                    'embedding': embedding,
                    'importance': concept_data.get('importance', 5.0),
                    'category': concept_data.get('category', 'general'),
                    'extraction_method': concept_data.get('extraction_method', 'llm_semantic'),
                    # Add anchor information for binding compatibility
                    'anchor_exact': concept_name,
                    'anchor_alias': concept_name
                }

                result_concepts.append(result_concept)
            
            print(f"LLM extraction produced {len(result_concepts)} concepts")
            return result_concepts[:max_concepts]
            
        except Exception as e:
            print(f"LLM concept extraction failed: {e}")
            print("Falling back to enhanced n-gram extraction")
            return self._extract_concepts_fallback(paper_text, max_concepts)
    
    def _enhance_concept_backpack(self, concept: str, paper_text: str, base_context: str) -> str:
        """Enhance LLM-provided context with paper excerpts."""
        # Use existing backpack generation logic
        paper_backpack = self._generate_context_backpack(concept, paper_text, target_tokens=300)
        
        # Combine LLM context with paper excerpts
        if base_context and paper_backpack:
            return f"{base_context}. Context from paper: {paper_backpack}"
        elif paper_backpack:
            return paper_backpack
        else:
            return base_context or f"Key concept: {concept}"


if __name__ == "__main__":
    # Test the concept extractor
    extractor = PaperConceptExtractor()
    
    # Test with sample text
    sample_text = """
    This paper introduces a novel approach to mechanistic interpretability using sparse autoencoders 
    and steering vectors. We demonstrate that activation patching can reveal how transformer models 
    process information through their neural network layers. Our method uses gradient descent 
    optimization to train concept bottleneck models that provide better feature visualization.
    The attention mechanism in these models shows interesting patterns when we apply reinforcement 
    learning techniques. We compare our approach to existing machine learning methods for 
    interpretability and show significant improvements in understanding how deep learning models work.
    """
    
    concepts = extractor.extract_concepts(sample_text)
    print(f"Extracted {len(concepts)} concepts:")
    for concept in concepts:
        print(f"- {concept['concept']}")
        print(f"  Context: {concept.get('backpack_m', concept.get('backpack', ''))[:100]}...")
        print()