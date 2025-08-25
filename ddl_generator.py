#!/usr/bin/env python3

import json
import time
import openai
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import aiohttp
from dataclasses import dataclass


@dataclass
class TestSpecification:
    """Structure for test specifications in daydream hypotheses."""
    type: str
    dataset_or_component: str
    manipulation: str
    metric: str
    expected_direction: str
    success_threshold: float
    timeframe_days: int


class DDLGenerator:
    """
    Daydream generator that creates speculative hypotheses connecting paper concepts to corpus concepts.
    Uses best-of-3 generation at different temperatures for quality.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio", model: str = "gpt-oss-120b"):
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = model
        self.temperatures = [0.2, 0.3, 0.5]  # Best-of-3 temperatures, lower first
        
        # Valid test types
        self.valid_test_types = {
            'ablation_control', 'negative_control', 'holdout_generalization', 
            'counterfactual_rewrite', 'adversarial_probe'
        }
        
        self.valid_directions = {'increase', 'decrease', 'no_change'}
    
    def generate_daydream(self, paper_concept: str, paper_backpack: str,
                         corpus_concept: str, corpus_snippet: str,
                         paper_anchor_exact: Optional[str] = None,
                         paper_anchor_alias: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a single daydream hypothesis using best-of-3 approach.
        
        Args:
            paper_concept: Name of concept from the paper
            paper_backpack: Context snippet from paper (100-250 tokens)
            corpus_concept: Name of concept from corpus
            corpus_snippet: Context snippet from corpus (30-80 tokens)
            paper_anchor_exact: Preferred exact anchor phrase from paper
            paper_anchor_alias: Normalized fallback anchor phrase
            
        Returns:
            Best hypothesis dict or None if all attempts fail
        """
        candidates = []
        
        # Generate candidates at different temperatures
        for temp in self.temperatures:
            try:
                hypothesis = self._generate_single_hypothesis(
                    paper_concept, paper_backpack, corpus_concept, corpus_snippet, temp,
                    paper_anchor_exact, paper_anchor_alias
                )
                if hypothesis:
                    # Apply autocorrect before validation
                    hypothesis = self._autocorrect_hypothesis(hypothesis)
                    if self._validate_hypothesis_structure(hypothesis):
                        candidates.append((hypothesis, temp))
            except Exception as e:
                print(f"Generation failed at temp {temp}: {e}")
                continue
        
        if not candidates:
            return None
        
        # Return the candidate from the lowest temperature that succeeded
        # (generally more reliable/coherent)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def _generate_single_hypothesis(self, paper_concept: str, paper_backpack: str,
                                  corpus_concept: str, corpus_snippet: str, 
                                  temperature: float,
                                  paper_anchor_exact: Optional[str] = None,
                                  paper_anchor_alias: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a single hypothesis at given temperature."""
        
        # Choose the best available anchor and make it explicit
        target_anchor = paper_anchor_exact or paper_anchor_alias or paper_concept
        
        # Explicit anchor requirement for deterministic inclusion
        anchor_requirement = f'You MUST include the exact phrase "{target_anchor}" verbatim as a quoted substring in your hypothesis. Do not paraphrase, alter spacing, case, or punctuation of this anchor phrase.'
        
        system_prompt = f"""You are generating a single, falsifiable hypothesis that bridges a paper claim and an external source.

CRITICAL ANCHOR REQUIREMENT:
{anchor_requirement}

RULES:
- You MUST include the paper concept "{paper_concept}" verbatim in the hypothesis text
- You MUST include the anchor phrase "{target_anchor}" exactly once as written above
- The hypothesis must be one paragraph (120-180 words) with this structure:
  CLAIM: <clear claim>
  MECHANISM: <why this could be true> 
  TEST: <how to check quickly>
  SIGNALS: <what observations would support or refute>
- Return only JSON between <json> and </json>

Schema:
{{
  "hypothesis": "120-180 words including paper concept AND anchor phrase \"{target_anchor}\" verbatim",
  "paper_anchor": "{target_anchor}",
  "test": {{
    "type": "ablation_control|negative_control|holdout_generalization|counterfactual_rewrite|adversarial_probe",
    "dataset_or_component": "specific dataset or model component to test",
    "manipulation": "what you would change/test",
    "metric": "how you would measure the result", 
    "expected_direction": "increase|decrease|no_change",
    "success_threshold": 0.0-1.0,
    "timeframe_days": 1-90
  }}
}}

IMPORTANT: Your hypothesis MUST contain the exact phrase \"{target_anchor}\" and your paper_anchor field MUST be exactly \"{target_anchor}\"."""
        
        user_prompt = f"""PAPER CONCEPT: {paper_concept}
PAPER CONTEXT (excerpt 100–250 tokens): {paper_backpack}

CORPUS CONCEPT: {corpus_concept}
CORPUS CONTEXT (1–2 sentences): {corpus_snippet}

REQUIRED ANCHOR TO COPY: "{target_anchor}"

Generate your hypothesis now. CRITICAL REQUIREMENTS:
1. Include "{paper_concept}" verbatim in the hypothesis
2. Include the exact phrase "{target_anchor}" verbatim in your hypothesis (copy it exactly as shown above)
3. Set paper_anchor field to exactly "{target_anchor}"

Return only the JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=1.0,
                max_tokens=2000,
                timeout=45
            )
            
            content = (response.choices[0].message.content or "").strip()
            
            # Use robust JSON repair
            hypothesis_data = self._json_repair(content)
            if hypothesis_data:
                # Post-generation anchor verification
                if self._verify_anchor_inclusion(hypothesis_data):
                    return hypothesis_data
                else:
                    # Single retry with explicit warning
                    print(f"Anchor missing in hypothesis at temp {temperature}, retrying...")
                    return self._retry_with_anchor_warning(paper_concept, paper_backpack, corpus_concept, corpus_snippet, temperature, paper_anchor_exact, paper_anchor_alias)
            else:
                print(f"JSON repair failed for temp {temperature}")
                print(f"Raw content: {content}")
                return None
            
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    def _verify_anchor_inclusion(self, hypothesis_data: Dict[str, Any]) -> bool:
        """Verify that paper_anchor appears verbatim in hypothesis text."""
        hypothesis = hypothesis_data.get('hypothesis', '')
        paper_anchor = hypothesis_data.get('paper_anchor', '')
        
        if not paper_anchor:
            return False
        
        # Normalize for comparison (handle quotes/dashes)
        def normalize_text(text):
            import re
            text = re.sub(r"[''`]", "'", text)  # Normalize quotes
            text = re.sub(r"[\u2013\u2014-]", "-", text)  # Normalize dashes
            return text.lower().strip()
        
        norm_hypothesis = normalize_text(hypothesis)
        norm_anchor = normalize_text(paper_anchor)
        
        return norm_anchor in norm_hypothesis
    
    def _retry_with_anchor_warning(self, paper_concept: str, paper_backpack: str,
                                 corpus_concept: str, corpus_snippet: str, 
                                 temperature: float,
                                 paper_anchor_exact: Optional[str] = None,
                                 paper_anchor_alias: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retry generation with explicit anchor warning."""
        
        # Choose the best available anchor for retry - same as main generation
        target_anchor = paper_anchor_exact or paper_anchor_alias or paper_concept
        
        # Even more explicit for retry
        anchor_requirement = f'CRITICAL: You MUST copy the exact phrase "{target_anchor}" verbatim into your hypothesis. Your previous attempt failed because this phrase was missing.'
        
        system_prompt = f"""CRITICAL: Your previous attempt failed because the anchor phrase did not appear in the hypothesis text.

You are generating a hypothesis that bridges a paper claim and external source.

{anchor_requirement}

RULES:
- Include "{paper_concept}" verbatim in the hypothesis
- Include the exact phrase "{target_anchor}" verbatim in your hypothesis (copy it exactly)
- Set paper_anchor field to exactly "{target_anchor}"
- 120-180 words total

Schema:
{{
  "hypothesis": "Must contain \"{target_anchor}\" verbatim",
  "paper_anchor": "{target_anchor}",
  "test": {{
    "type": "ablation_control|negative_control|holdout_generalization|counterfactual_rewrite|adversarial_probe",
    "dataset_or_component": "specific dataset or model component",
    "manipulation": "what to change/test",
    "metric": "measurement method",
    "expected_direction": "increase|decrease|no_change",
    "success_threshold": 0.0-1.0,
    "timeframe_days": 1-90
  }}
}}

Return only JSON between <json> and </json>."""
        
        user_prompt = f"""PAPER CONCEPT: {paper_concept}
PAPER CONTEXT: {paper_backpack}

CORPUS CONCEPT: {corpus_concept}
CORPUS CONTEXT: {corpus_snippet}

REQUIRED ANCHOR TO COPY: "{target_anchor}"

FIX THE ERROR: Include the exact phrase "{target_anchor}" verbatim in your hypothesis AND set paper_anchor to exactly "{target_anchor}".

Return only the JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=1.0,
                max_tokens=2000,
                timeout=45
            )
            
            content = (response.choices[0].message.content or "").strip()
            hypothesis_data = self._json_repair(content)
            
            if hypothesis_data:
                # Mark as retry attempt
                hypothesis_data['anchor_retry'] = True
                return hypothesis_data
            else:
                return None
                
        except Exception as e:
            print(f"Retry API call failed: {e}")
            return None
    
    def _word_count(self, s: str) -> int:
        """Count words in string using regex."""
        import re
        return len(re.findall(r"\b\w+\b", s))
    
    def _extract_first_json(self, s: str) -> Optional[str]:
        """Extract first balanced JSON object from string."""
        depth = 0
        start = None
        for i, ch in enumerate(s):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start:i+1]
        return None
    
    def _json_repair(self, s: str) -> Optional[Dict[str, Any]]:
        """Robust JSON extraction and repair for local LLM outputs."""
        # Strip common wrapper patterns
        s = s.strip()
        s = s.replace("```json", "").replace("```", "")
        
        # Handle incomplete <json> tags - append </json> if missing
        if "<json>" in s and "</json>" not in s:
            s = s + "</json>"
        
        s = s.replace("<json>", "").replace("</json>", "")
        
        # Try to complete incomplete JSON by counting braces
        if s.count('{') > s.count('}'):
            missing_closes = s.count('{') - s.count('}')
            s = s + '}' * missing_closes
        
        # Extract balanced JSON if surrounded by prose
        candidate = self._extract_first_json(s) or s
        
        # Fix common newline issues ONLY inside string values, not JSON structure
        # Don't replace structural newlines - only escape newlines within string values
        
        # Try fast path first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # Apply heuristic fixes for common local LLM mistakes
        import re
        # Replace single quotes with double quotes (carefully)
        candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
        # Remove trailing commas
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate2)
        
        try:
            return json.loads(candidate2)
        except:
            return None
    
    def _autocorrect_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Post-validation autocorrect for common issues."""
        corrected = hypothesis.copy()
        
        # Fix word count padding for 70-79 word hypotheses
        if 'hypothesis' in corrected:
            hyp_text = corrected['hypothesis']
            wc = self._word_count(hyp_text)
            if 70 <= wc < 80:
                hyp_text = (hyp_text.rstrip(". ") + ". This motivates a concrete test.").strip()
                corrected['hypothesis'] = hyp_text
        
        # Replace paper concept token placeholder
        if 'hypothesis' in corrected:
            # Find and replace <<concept>> pattern with plain concept
            import re
            hyp_text = corrected['hypothesis']
            # Replace any <<...>> pattern with the content inside
            hyp_text = re.sub(r'<<([^>]+)>>', r'\1', hyp_text)
            corrected['hypothesis'] = hyp_text
        
        # Ensure test exists
        if 'test' in corrected and isinstance(corrected['test'], dict):
            test = corrected['test']
            
            # Clamp success_threshold to [0,1]
            if 'success_threshold' in test:
                try:
                    threshold = float(test['success_threshold'])
                    test['success_threshold'] = max(0.0, min(1.0, threshold))
                except (ValueError, TypeError):
                    test['success_threshold'] = 0.7  # Safe default
            
            # Clamp timeframe_days to [1,90]
            if 'timeframe_days' in test:
                try:
                    days = int(test['timeframe_days'])
                    test['timeframe_days'] = max(1, min(90, days))
                except (ValueError, TypeError):
                    test['timeframe_days'] = 14  # Safe default
            
            # Normalize expected_direction
            if 'expected_direction' in test:
                direction = str(test['expected_direction']).lower()
                direction_map = {
                    'inc': 'increase', 'incr': 'increase', 'up': 'increase',
                    'dec': 'decrease', 'decr': 'decrease', 'down': 'decrease',
                    'same': 'no_change', 'unchanged': 'no_change', 'stable': 'no_change'
                }
                test['expected_direction'] = direction_map.get(direction, direction)
        
        # Ensure paper_anchor is reasonable length
        if 'paper_anchor' in corrected:
            anchor = str(corrected['paper_anchor'])
            if len(anchor) > 80:
                corrected['paper_anchor'] = anchor[:80].strip()
        
        return corrected
    
    def _validate_hypothesis_structure(self, hypothesis: Dict[str, Any]) -> bool:
        """Validate that hypothesis has required structure and valid values."""
        required_keys = {'hypothesis', 'paper_anchor', 'test'}
        if not all(key in hypothesis for key in required_keys):
            return False
        
        # Validate hypothesis text - FIX: use word count, more flexible range
        hyp_text = hypothesis.get('hypothesis', '')
        wc = self._word_count(hyp_text)
        if not (80 <= wc <= 200):  # More flexible for local LLM
            print(f"Word count {wc} outside range 80-200")
            return False
        
        # Validate paper_anchor
        if not isinstance(hypothesis.get('paper_anchor'), str):
            return False
        
        # Validate test structure
        test = hypothesis.get('test', {})
        required_test_keys = {
            'type', 'dataset_or_component', 'manipulation', 
            'metric', 'expected_direction', 'success_threshold', 'timeframe_days'
        }
        
        if not all(key in test for key in required_test_keys):
            return False
        
        # Validate test type
        if test['type'] not in self.valid_test_types:
            return False
        
        # Validate direction
        if test['expected_direction'] not in self.valid_directions:
            return False
        
        # Validate threshold and timeframe
        try:
            threshold = float(test['success_threshold'])
            if not (0.0 <= threshold <= 1.0):
                return False
            
            timeframe = int(test['timeframe_days'])
            if not (1 <= timeframe <= 90):
                return False
                
        except (ValueError, TypeError):
            return False
        
        return True
    
    def generate_batch(self, concept_pairs: List[Dict[str, Any]], 
                      batch_size: int = 10, parallel_batches: int = 1) -> List[Dict[str, Any]]:
        """
        Generate daydreams for a batch of concept pairs.
        
        Args:
            concept_pairs: List of pair dictionaries from sampler
            batch_size: Number of pairs to process simultaneously
            parallel_batches: Number of batches to run in parallel
            
        Returns:
            List of successful daydream generations
        """
        if parallel_batches <= 1:
            return self._generate_batch_sequential(concept_pairs, batch_size)
        else:
            return self._generate_batch_parallel(concept_pairs, batch_size, parallel_batches)
    
    def _generate_batch_sequential(self, concept_pairs: List[Dict[str, Any]], 
                                 batch_size: int = 10) -> List[Dict[str, Any]]:
        """Sequential batch processing (original method)."""
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(concept_pairs), batch_size):
            batch = concept_pairs[i:i+batch_size]
            batch_results = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(concept_pairs)-1)//batch_size + 1}")
            
            for pair in batch:
                daydream = self.generate_daydream(
                    paper_concept=pair['paper_concept'],
                    paper_backpack=pair['paper_backpack'],
                    corpus_concept=pair['corpus_concept'], 
                    corpus_snippet=pair['corpus_snippet'],
                    paper_anchor_exact=pair.get('paper_anchor_exact'),
                    paper_anchor_alias=pair.get('paper_anchor_alias')
                )
                
                if daydream:
                    # Add metadata from the pair
                    daydream_with_meta = {
                        'pair': [pair['paper_concept'], pair['corpus_concept']],
                        'paper_concept': pair['paper_concept'],
                        'corpus_concept': pair['corpus_concept'],
                        'author': pair['author'],
                        'year': pair['year'],
                        'title': pair['title'],
                        'doc_id': pair['doc_id'],
                        'sampling_score': pair['score'],
                        **daydream
                    }
                    batch_results.append(daydream_with_meta)
                
                # Brief pause to avoid overwhelming the API
                time.sleep(0.1)
            
            results.extend(batch_results)
            
            # Brief pause between batches
            if i + batch_size < len(concept_pairs):
                time.sleep(1)
        
        return results
    
    def _generate_batch_parallel(self, concept_pairs: List[Dict[str, Any]], 
                               batch_size: int, parallel_batches: int) -> List[Dict[str, Any]]:
        """Parallel batch processing using threading."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Split pairs into chunks for parallel processing
        chunk_size = max(1, len(concept_pairs) // parallel_batches)
        chunks = [concept_pairs[i:i+chunk_size] for i in range(0, len(concept_pairs), chunk_size)]
        
        print(f"Processing {len(concept_pairs)} pairs in {len(chunks)} parallel chunks")
        
        all_results = []
        
        def process_chunk(chunk_pairs, chunk_id):
            chunk_results = []
            for pair in chunk_pairs:
                daydream = self.generate_daydream(
                    paper_concept=pair['paper_concept'],
                    paper_backpack=pair['paper_backpack'],
                    corpus_concept=pair['corpus_concept'], 
                    corpus_snippet=pair['corpus_snippet'],
                    paper_anchor_exact=pair.get('paper_anchor_exact'),
                    paper_anchor_alias=pair.get('paper_anchor_alias')
                )
                
                if daydream:
                    daydream_with_meta = {
                        'pair': [pair['paper_concept'], pair['corpus_concept']],
                        'paper_concept': pair['paper_concept'],
                        'corpus_concept': pair['corpus_concept'],
                        'author': pair['author'],
                        'year': pair['year'],
                        'title': pair['title'],
                        'doc_id': pair['doc_id'],
                        'sampling_score': pair['score'],
                        **daydream
                    }
                    chunk_results.append(daydream_with_meta)
                
                time.sleep(0.05)  # Shorter pause for parallel processing
            
            return chunk_results
        
        # Execute chunks in parallel
        with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, i): i 
                             for i, chunk in enumerate(chunks)}
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    print(f"Completed chunk {chunk_id + 1}/{len(chunks)}: {len(chunk_results)} daydreams")
                except Exception as e:
                    print(f"Chunk {chunk_id} failed: {e}")
        
        return all_results
    
    def save_generated_daydreams(self, daydreams: List[Dict[str, Any]], output_path: str):
        """Save generated daydreams to JSONL file."""
        with open(output_path, 'w') as f:
            for daydream in daydreams:
                f.write(json.dumps(daydream) + '\n')
    
    def load_generated_daydreams(self, input_path: str) -> List[Dict[str, Any]]:
        """Load generated daydreams from JSONL file."""
        daydreams = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    daydreams.append(json.loads(line))
        return daydreams


if __name__ == "__main__":
    # Test the generator
    generator = DDLGenerator()
    
    # Mock concept pair
    test_pair = {
        'paper_concept': 'sparse autoencoder',
        'paper_backpack': 'Sparse autoencoders learn compressed representations by forcing most neurons to be inactive, creating interpretable features that can be used for steering model behavior.',
        'corpus_concept': 'attention mechanism',
        'corpus_snippet': 'Attention mechanisms allow models to selectively focus on relevant parts of the input sequence.',
        'author': 'Test Author',
        'year': 2023,
        'title': 'Test Paper',
        'doc_id': 'test123',
        'score': 0.75
    }
    
    print("Testing daydream generation...")
    result = generator.generate_daydream(
        test_pair['paper_concept'],
        test_pair['paper_backpack'],
        test_pair['corpus_concept'],
        test_pair['corpus_snippet']
    )
    
    if result:
        print("Generated daydream:")
        print(json.dumps(result, indent=2))
    else:
        print("Generation failed")