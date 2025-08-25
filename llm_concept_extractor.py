#!/usr/bin/env python3

import json
import re
import openai
from typing import Dict, List, Any, Optional
from collections import defaultdict

class LLMConceptExtractor:
    """
    LLM-based concept extraction that uses semantic understanding
    instead of n-gram pattern matching for better domain relevance.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1", api_key: str = "lm-studio", model: str = "gpt-oss-120b"):
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = model
        
        # Domain-specific guidance for AI safety/ML papers
        self.domain_terms = {
            'ai_safety': [
                'alignment', 'interpretability', 'safety', 'robustness', 'adversarial',
                'backdoor', 'poisoning', 'steering', 'control', 'oversight'
            ],
            'ml_methods': [
                'autoencoder', 'transformer', 'attention', 'embedding', 'gradient',
                'optimization', 'regularization', 'activation', 'layer', 'network'
            ],
            'research_methods': [
                'evaluation', 'benchmark', 'metric', 'dataset', 'experiment',
                'analysis', 'study', 'methodology', 'approach', 'framework'
            ]
        }

    def _make_api_call(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """Make API call to local LLM using OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return None

    def extract_concepts_llm(self, paper_text: str, max_concepts: int = 30) -> List[Dict[str, Any]]:
        """
        Extract concepts using LLM semantic understanding
        """
        # Split paper into sections for better processing
        sections = self._split_paper_sections(paper_text)
        
        # Extract concepts from each section with different priorities
        all_concepts = []
        
        # Title/Abstract - highest priority
        if sections.get('title') or sections.get('abstract'):
            title_abstract = f"{sections.get('title', '')}\n\n{sections.get('abstract', '')}"
            concepts = self._extract_from_section(title_abstract, section_type="title_abstract", weight=3.0)
            all_concepts.extend(concepts)
        
        # Introduction - high priority for context
        if sections.get('introduction'):
            concepts = self._extract_from_section(sections['introduction'], section_type="introduction", weight=2.0)
            all_concepts.extend(concepts)
        
        # Methods/Body - medium priority for technical terms
        body_text = sections.get('body', paper_text)
        if body_text and len(body_text) > 500:  # Only if substantial content
            # Sample key paragraphs to avoid token limits
            key_paragraphs = self._sample_key_paragraphs(body_text, max_paragraphs=5)
            concepts = self._extract_from_section(key_paragraphs, section_type="methods", weight=1.0)
            all_concepts.extend(concepts)
        
        # Deduplicate and rank concepts
        ranked_concepts = self._rank_and_deduplicate(all_concepts, max_concepts)
        
        return ranked_concepts

    def _split_paper_sections(self, paper_text: str) -> Dict[str, str]:
        """Split paper into logical sections"""
        sections = {}
        
        # Extract title (first line or lines in caps/title case)
        lines = paper_text.split('\n')
        potential_title = []
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (line.isupper() or len(line.split()) <= 15):
                potential_title.append(line)
            elif potential_title:
                break
        
        if potential_title:
            sections['title'] = '\n'.join(potential_title)
        
        # Look for abstract section
        abstract_match = re.search(r'\bAbstract\b[:\s]*\n(.*?)(?=\n\s*\n|\n[A-Z]|\n\d+\.)', paper_text, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()
        
        # Look for introduction
        intro_match = re.search(r'\b(?:Introduction|1\.?\s*Introduction)\b[:\s]*\n(.*?)(?=\n\s*\n[A-Z]|\n\d+\.)', paper_text, re.DOTALL | re.IGNORECASE)
        if intro_match:
            sections['introduction'] = intro_match.group(1).strip()
        
        # Everything else as body
        sections['body'] = paper_text
        
        return sections

    def _sample_key_paragraphs(self, text: str, max_paragraphs: int = 5) -> str:
        """Sample key paragraphs that likely contain important concepts"""
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        # Score paragraphs by technical term density
        scored_paragraphs = []
        for para in paragraphs:
            score = 0
            for category, terms in self.domain_terms.items():
                for term in terms:
                    score += para.lower().count(term.lower())
            scored_paragraphs.append((score, para))
        
        # Take top scoring paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        selected = [para for score, para in scored_paragraphs[:max_paragraphs]]
        
        return '\n\n'.join(selected)

    def _extract_from_section(self, section_text: str, section_type: str, weight: float) -> List[Dict[str, Any]]:
        """Extract concepts from a specific section using LLM"""
        
        # Create section-specific prompt
        if section_type == "title_abstract":
            prompt = self._create_title_abstract_prompt(section_text)
        elif section_type == "introduction":
            prompt = self._create_introduction_prompt(section_text)
        else:
            prompt = self._create_methods_prompt(section_text)
        
        # Get LLM response
        response = self._make_api_call(prompt, max_tokens=800, temperature=0.1)
        if not response:
            return []
        
        # Parse concepts from response
        concepts = self._parse_concept_response(response, section_type, weight)
        return concepts

    def _create_title_abstract_prompt(self, text: str) -> str:
        return f"""Analyze this paper title and abstract to identify the most important technical concepts, methods, and domain-specific terms. Focus on:

1. Key technical methods or algorithms
2. Important domain concepts (AI safety, ML interpretability, etc.)
3. Novel contributions or techniques
4. Specific model types or architectures

Text to analyze:
{text}

Extract up to 10 concepts as a JSON list. For each concept, provide:
- "term": the exact phrase (prefer multi-word technical terms)
- "importance": score 1-10 based on centrality to the paper
- "category": one of [method, concept, model, technique, domain_term]
- "context": brief explanation of why this term is important

Format as valid JSON array only, no other text."""

    def _create_introduction_prompt(self, text: str) -> str:
        return f"""Analyze this introduction section to identify key background concepts and problem definitions. Focus on:

1. Problem domain terminology
2. Related work concepts being referenced
3. Technical background terms
4. Research area definitions

Text to analyze:
{text[:2000]}  

Extract up to 8 concepts as JSON. For each:
- "term": exact phrase (prefer established technical terms)
- "importance": score 1-10 
- "category": one of [background, problem, domain, related_work]
- "context": brief explanation

Format as valid JSON array only."""

    def _create_methods_prompt(self, text: str) -> str:
        return f"""Analyze these key paragraphs to identify specific technical methods and implementation details. Focus on:

1. Algorithm names and techniques
2. Model architectures or components  
3. Evaluation metrics or methods
4. Technical parameters or settings

Text to analyze:
{text[:2000]}

Extract up to 6 concepts as JSON. For each:
- "term": exact technical phrase
- "importance": score 1-10
- "category": one of [algorithm, metric, parameter, implementation]
- "context": brief technical explanation

Format as valid JSON array only."""

    def _parse_concept_response(self, response: str, section_type: str, weight: float) -> List[Dict[str, Any]]:
        """Parse LLM response into concept objects"""
        concepts = []
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                concept_list = json.loads(json_match.group(0))
                
                for item in concept_list:
                    if isinstance(item, dict) and 'term' in item:
                        concept = {
                            'name': item['term'].strip(),
                            'importance': item.get('importance', 5) * weight,
                            'category': item.get('category', 'general'),
                            'backpack': item.get('context', f"Key term from {section_type}"),
                            'section': section_type,
                            'extraction_method': 'llm_semantic'
                        }
                        concepts.append(concept)
                        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM concept response: {e}")
            # Fallback: extract terms from response text
            terms = re.findall(r'"([^"]+)"', response)
            for term in terms[:10]:  # Limit fallback terms
                if len(term.split()) >= 2 and len(term) > 5:  # Prefer multi-word terms
                    concept = {
                        'name': term,
                        'importance': 5.0 * weight,
                        'category': 'general',
                        'backpack': f"Extracted from {section_type}",
                        'section': section_type,
                        'extraction_method': 'llm_fallback'
                    }
                    concepts.append(concept)
        
        return concepts

    def _rank_and_deduplicate(self, concepts: List[Dict[str, Any]], max_concepts: int) -> List[Dict[str, Any]]:
        """Rank concepts by importance and remove duplicates"""
        
        # Deduplicate by normalized name
        seen_terms = set()
        unique_concepts = []
        
        for concept in concepts:
            normalized = concept['name'].lower().strip()
            if normalized not in seen_terms and len(normalized) > 3:
                seen_terms.add(normalized)
                unique_concepts.append(concept)
        
        # Sort by importance score
        unique_concepts.sort(key=lambda x: x['importance'], reverse=True)
        
        # Take top N concepts
        final_concepts = unique_concepts[:max_concepts]
        
        return final_concepts

# Integration function for existing pipeline
def create_llm_enhanced_extractor(config: Dict[str, Any]) -> 'LLMConceptExtractor':
    """Factory function to create LLM-based extractor with config"""
    return LLMConceptExtractor(
        api_base=config.get('api_base', 'http://localhost:1234/v1'),
        api_key=config.get('api_key', 'lm-studio'),
        model=config.get('model', 'gpt-oss-120b')
    )