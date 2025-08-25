#!/usr/bin/env python3

import json
import re
import openai
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CriticScores:
    """Structure for critic evaluation scores."""
    novelty: float
    coherence: float
    usefulness: float
    binding: float
    valid: bool
    reasons: List[str]


class DDLCritic:
    """
    Critic for evaluating and filtering generated daydream hypotheses.
    Validates JSON schema and scores on multiple dimensions.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio", model: str = "gpt-oss-120b"):
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = model
        
        # Validation thresholds (relaxed for local LLM testing)
        self.min_binding_score = 0.15  # Two-tier screening: lower threshold
        self.provisional_binding_score = 0.3  # Original threshold for premium tier
        self.min_overall_score = 0.2
        self.min_coherence_score = 0.3
        
        # Valid test types and directions
        self.valid_test_types = {
            'ablation_control', 'negative_control', 'holdout_generalization', 
            'counterfactual_rewrite', 'adversarial_probe'
        }
        self.valid_directions = {'increase', 'decrease', 'no_change'}
    
    def _word_count(self, s: str) -> int:
        """Count words in string using regex."""
        import re
        return len(re.findall(r"\b\w+\b", s))
    
    def _extract_first_json(self, s: str) -> Optional[str]:
        """Extract first balanced JSON object from string."""
        depth = 0
        start = None
        for i, ch in enumerate(s or ""):
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
        """Robust JSON extraction and repair for critic outputs."""
        s = (s or "").strip().replace("```json", "").replace("```", "")
        s = s.replace("<json>", "").replace("</json>", "")
        candidate = self._extract_first_json(s) or s
        
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            import re
            candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
            candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate2)
            try:
                return json.loads(candidate2)
            except:
                return None
    
    def _hard_binding(self, daydream: Dict[str, Any]) -> bool:
        """Deterministic check if hypothesis explicitly references paper concept."""
        hyp = (daydream.get('hypothesis') or "").lower()
        anchor = (daydream.get('paper_anchor') or "").lower().strip()
        if not anchor:
            return False
        
        # Exact containment
        if anchor in hyp:
            return True
        
        # Fuzzy containment: ignore punctuation, collapse spaces
        import re
        a = re.sub(r'\W+', ' ', anchor).strip()
        h = re.sub(r'\W+', ' ', hyp)
        return a and a in h
    
    def _soft_binding(self, daydream: Dict[str, Any]) -> float:
        """Soft binding score using token Jaccard + sequence similarity."""
        hypothesis = (daydream.get('hypothesis') or "").strip()
        paper_anchor = (daydream.get('paper_anchor') or "").strip()
        
        # Handle empty anchor
        if not paper_anchor or not hypothesis:
            return 0.0
        
        import re
        from difflib import SequenceMatcher
        
        # Tokenize: remove non-alphanumeric, split, filter short tokens
        def tokenize(text):
            return [t for t in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split() if len(t) > 2]
        
        H = set(tokenize(hypothesis))
        P = set(tokenize(paper_anchor))
        
        # Token Jaccard similarity
        if not H or not P:
            jaccard = 0.0
        else:
            jaccard = len(H & P) / len(H | P)
        
        # Sequence similarity for phrase-level matching
        seq_ratio = SequenceMatcher(None, hypothesis.lower(), paper_anchor.lower()).ratio()
        
        # Return max of both approaches
        return max(jaccard, seq_ratio)
    
    def _normalize_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize test fields to handle format drift."""
        t = dict(test)
        
        # Type normalization
        t['type'] = str(t.get('type', '')).lower().strip()
        aliases = {
            "ablation": "ablation_control",
            "neg_control": "negative_control", 
            "holdout": "holdout_generalization",
            "counterfactual": "counterfactual_rewrite",
            "adv_probe": "adversarial_probe"
        }
        t['type'] = aliases.get(t['type'], t['type'])
        
        # Direction normalization
        direction = str(t.get('expected_direction', '')).lower().strip()
        dir_map = {
            "inc": "increase", "incr": "increase", "up": "increase",
            "dec": "decrease", "decr": "decrease", "down": "decrease", 
            "unchanged": "no_change", "nochange": "no_change", "stable": "no_change"
        }
        t['expected_direction'] = dir_map.get(direction, direction)
        
        # Clamp numerics
        try:
            t['success_threshold'] = float(t.get('success_threshold', 0.0))
        except:
            t['success_threshold'] = 0.0
        t['success_threshold'] = max(0.0, min(1.0, t['success_threshold']))
        
        try:
            t['timeframe_days'] = int(t.get('timeframe_days', 14))
        except:
            t['timeframe_days'] = 14
        t['timeframe_days'] = max(1, min(90, t['timeframe_days']))
        
        return t
    
    def _heuristic_scores(self, daydream: Dict[str, Any]) -> Dict[str, float]:
        """Fast heuristic scoring fallback when LLM critic fails."""
        hyp = daydream.get('hypothesis') or ""
        wc = self._word_count(hyp)
        binding = 1.0 if self._hard_binding(daydream) else 0.0
        
        # Rough novelty proxy: encourage specificity
        import re
        numerics = len(re.findall(r"\d", hyp))
        connectors = len(re.findall(r"\b(because|therefore|hence|so that|in order to)\b", hyp.lower()))
        hedges = len(re.findall(r"\b(might|could|perhaps|possibly|may)\b", hyp.lower()))
        
        novelty = min(1.0, 0.2 + 0.1*numerics + 0.05*connectors - 0.04*hedges)
        coherence = min(1.0, 0.3 + 0.01*wc + 0.1*connectors - 0.02*hedges)
        
        # Usefulness based on test completeness
        usefulness = 0.0
        test = daydream.get('test', {})
        if all(k in test for k in ['type', 'dataset_or_component', 'manipulation', 
                                  'metric', 'expected_direction', 'success_threshold', 'timeframe_days']):
            usefulness = 0.6 + 0.1*bool(numerics) + 0.1*bool(connectors)
        
        return {
            "novelty": max(0.0, min(1.0, novelty)),
            "coherence": max(0.0, min(1.0, coherence)), 
            "usefulness": max(0.0, min(1.0, usefulness)),
            "binding": binding
        }
    
    def evaluate_daydream(self, daydream: Dict[str, Any]) -> CriticScores:
        """
        Evaluate a single daydream hypothesis.
        
        Args:
            daydream: Generated daydream with hypothesis, paper_anchor, and test
            
        Returns:
            CriticScores with validation status and dimensional scores
        """
        # Normalize test fields before validation
        if 'test' in daydream and isinstance(daydream['test'], dict):
            daydream['test'] = self._normalize_test(daydream['test'])
        
        # First, validate schema
        schema_valid, schema_reasons = self._validate_schema(daydream)
        if not schema_valid:
            return CriticScores(
                novelty=0.0, coherence=0.0, usefulness=0.0, binding=0.0,
                valid=False, reasons=schema_reasons
            )
        
        # Get AI critic scores
        ai_scores = self._get_ai_critic_scores(daydream)
        
        if ai_scores is None:
            return CriticScores(
                novelty=0.0, coherence=0.0, usefulness=0.0, binding=0.0,
                valid=False, reasons=["Failed to get AI critic evaluation"]
            )
        
        # Enhanced binding check with soft matching and diagnostics
        hard_binding = self._hard_binding(daydream)
        soft_binding = self._soft_binding(daydream)
        
        # Calculate binding score: hard pass=1.0, else soft scoring
        if hard_binding:
            binding_score = 1.0
        elif soft_binding >= 0.6:
            binding_score = 0.25 + 0.5 * soft_binding  # 0.55-0.75 range
        else:
            binding_score = 0.0
        
        # Override AI critic binding with our computed score
        ai_scores['binding'] = binding_score
        
        # Add diagnostics for debugging
        diagnostics = {
            "hard": hard_binding,
            "soft": round(soft_binding, 3),
            "binding_score": round(binding_score, 3)
        }
        
        # Validate thresholds with two-tier screening
        reasons = []
        valid = True
        
        if binding_score < self.min_binding_score:
            valid = False
            reasons.append(f"Binding score {binding_score:.2f} below threshold {self.min_binding_score}")
        
        # Add tier classification for accepted daydreams
        if valid:
            if hard_binding and binding_score >= self.provisional_binding_score:
                reasons.append("Premium tier: hard binding + high confidence")
            elif binding_score >= self.min_binding_score:
                reasons.append(f"Provisional tier: soft binding (score={binding_score:.2f})")
        
        if ai_scores['coherence'] < self.min_coherence_score:
            valid = False
            reasons.append(f"Coherence {ai_scores['coherence']:.2f} below {self.min_coherence_score} floor")
        
        overall_score = ai_scores['novelty'] * ai_scores['usefulness']
        
        if overall_score < self.min_overall_score:
            valid = False
            reasons.append(f"Overall score {overall_score:.2f} below threshold {self.min_overall_score}")
        
        # Store diagnostics for later use
        self._last_diagnostics = diagnostics
        
        return CriticScores(
            novelty=ai_scores['novelty'],
            coherence=ai_scores['coherence'],
            usefulness=ai_scores['usefulness'],
            binding=ai_scores['binding'],
            valid=valid,
            reasons=reasons
        )
    
    def _validate_schema(self, daydream: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate JSON schema structure."""
        reasons = []
        
        # Check required top-level keys
        required_keys = {'hypothesis', 'paper_anchor', 'test'}
        missing_keys = required_keys - set(daydream.keys())
        if missing_keys:
            reasons.append(f"Missing keys: {missing_keys}")
        
        # Validate hypothesis
        hypothesis = daydream.get('hypothesis', '')
        if not isinstance(hypothesis, str):
            reasons.append("Hypothesis must be a string")
        else:
            # Check word count instead of character count
            import re
            word_count = len(re.findall(r"\b\w+\b", hypothesis))
            if not (80 <= word_count <= 200):  # Flexible word count range
                reasons.append(f"Hypothesis word count {word_count} not in range 80-200")
        
        # Validate paper_anchor
        paper_anchor = daydream.get('paper_anchor', '')
        if not isinstance(paper_anchor, str):
            reasons.append("Paper anchor must be a string")
        elif len(paper_anchor.strip()) == 0:
            reasons.append("Paper anchor cannot be empty")
        
        # Validate test structure
        test = daydream.get('test', {})
        if not isinstance(test, dict):
            reasons.append("Test must be a dictionary")
        else:
            test_reasons = self._validate_test_schema(test)
            reasons.extend(test_reasons)
        
        return len(reasons) == 0, reasons
    
    def _validate_test_schema(self, test: Dict[str, Any]) -> List[str]:
        """Validate test specification schema."""
        reasons = []
        
        # Required test fields
        required_fields = {
            'type', 'dataset_or_component', 'manipulation', 
            'metric', 'expected_direction', 'success_threshold', 'timeframe_days'
        }
        
        missing_fields = required_fields - set(test.keys())
        if missing_fields:
            reasons.append(f"Missing test fields: {missing_fields}")
            return reasons  # Can't validate further without fields
        
        # Validate test type
        if test['type'] not in self.valid_test_types:
            reasons.append(f"Invalid test type '{test['type']}'. Must be one of {self.valid_test_types}")
        
        # Validate string fields are non-empty
        string_fields = ['dataset_or_component', 'manipulation', 'metric']
        for field in string_fields:
            value = test.get(field, '')
            if not isinstance(value, str) or len(value.strip()) == 0:
                reasons.append(f"Test field '{field}' must be a non-empty string")
        
        # Validate expected_direction
        if test['expected_direction'] not in self.valid_directions:
            reasons.append(f"Invalid expected_direction '{test['expected_direction']}'. Must be one of {self.valid_directions}")
        
        # Validate success_threshold
        try:
            threshold = float(test['success_threshold'])
            if not (0.0 <= threshold <= 1.0):
                reasons.append(f"Success threshold {threshold} must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            reasons.append("Success threshold must be a valid float between 0.0 and 1.0")
        
        # Validate timeframe_days
        try:
            timeframe = int(test['timeframe_days'])
            if not (1 <= timeframe <= 90):
                reasons.append(f"Timeframe {timeframe} must be between 1 and 90 days")
        except (ValueError, TypeError):
            reasons.append("Timeframe days must be a valid integer between 1 and 90")
        
        return reasons
    
    def _get_ai_critic_scores(self, daydream: Dict[str, Any]) -> Dict[str, float]:
        """Get AI critic scores using panel/median approach with robust JSON handling."""
        
        system_prompt = """You are an evaluator. Score ONLY what is present in the provided JSON fields.
Do not add external knowledge. Return a single JSON object between <json> and </json>:

{"valid": true/false, "reasons": ["..."], "scores": {"novelty": X, "coherence": Y, "usefulness": Z, "binding": W}}

Scoring guide:
- novelty: 0.0–1.0 (unexpectedness of the PAPER↔CORPUS link)
- coherence: 0.0–1.0 (internal logic; contradictions reduce score)
- usefulness: 0.0–1.0 (is the TEST specific, measurable, decision-relevant?)
- binding: 0.0–1.0 (hypothesis explicitly references PAPER concept and connects to CORPUS concept)"""
        
        user_prompt = f"""Evaluate this generated hypothesis:

HYPOTHESIS: {daydream.get('hypothesis', '')}

PAPER ANCHOR: {daydream.get('paper_anchor', '')}

TEST SPECIFICATION:
{json.dumps(daydream.get('test', {}), indent=2)}

Return JSON with valid, reasons, and scores."""
        
        # Panel approach: run two passes and median the scores
        scores = []
        for run in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nReturn only JSON between <json> and </json>."}
                    ],
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=1500,  # Much more generous for critic evaluation
                    timeout=30
                )
                
                content = (response.choices[0].message.content or "").strip()
                evaluation = self._json_repair(content)
                
                if evaluation and 'scores' in evaluation:
                    s = evaluation['scores']
                    if all(k in s for k in ['novelty', 'coherence', 'usefulness', 'binding']):
                        try:
                            parsed_scores = {k: float(s[k]) for k in ['novelty', 'coherence', 'usefulness', 'binding']}
                            # Validate ranges
                            if all(0.0 <= v <= 1.0 for v in parsed_scores.values()):
                                scores.append(parsed_scores)
                        except (ValueError, TypeError):
                            pass
                            
            except Exception as e:
                print(f"Critic run {run+1} failed: {e}")
                continue
        
        # If no valid scores from AI, use heuristics
        if not scores:
            print("AI critic failed, using heuristic fallback")
            return self._heuristic_scores(daydream)
        
        # Median of successful runs
        if len(scores) == 1:
            return scores[0]
        
        # Median of two runs
        import statistics as st
        keys = ['novelty', 'coherence', 'usefulness', 'binding']
        result = {k: st.median([s[k] for s in scores]) for k in keys}
        
        # Fill missing scores with heuristics if needed
        heuristic = self._heuristic_scores(daydream)
        for k in keys:
            if k not in result:
                result[k] = heuristic[k]
        
        return result
    
    def filter_daydreams(self, daydreams: List[Dict[str, Any]], 
                        keep_top: int = 100) -> List[Dict[str, Any]]:
        """
        Filter and rank daydreams, keeping only the best ones.
        
        Args:
            daydreams: List of generated daydreams
            keep_top: Maximum number of daydreams to keep
            
        Returns:
            Filtered list of daydreams with critic scores
        """
        scored_daydreams = []
        
        print(f"Evaluating {len(daydreams)} daydreams...")
        
        for i, daydream in enumerate(daydreams):
            if i % 10 == 0:
                print(f"Evaluated {i}/{len(daydreams)}")
            
            critic_scores = self.evaluate_daydream(daydream)
            
            if critic_scores.valid:
                # Add critic scores to daydream
                daydream_with_scores = daydream.copy()
                daydream_with_scores['critic'] = {
                    'novelty': critic_scores.novelty,
                    'coherence': critic_scores.coherence,
                    'usefulness': critic_scores.usefulness,
                    'binding': critic_scores.binding,
                    'overall_score': critic_scores.novelty * critic_scores.usefulness,  # Primary ranking metric
                    'diagnostics': getattr(self, '_last_diagnostics', {})
                }
                scored_daydreams.append(daydream_with_scores)
            else:
                # Show diagnostics for rejected daydreams too
                diagnostics = getattr(self, '_last_diagnostics', {})
                diagnostic_info = f" (hard={diagnostics.get('hard', 'N/A')}, soft={diagnostics.get('soft', 'N/A')})"
                print(f"Rejected daydream: {critic_scores.reasons}{diagnostic_info}")
        
        print(f"Accepted {len(scored_daydreams)}/{len(daydreams)} daydreams")
        
        # Sort by overall score (novelty * usefulness) and take top
        scored_daydreams.sort(key=lambda x: x['critic']['overall_score'], reverse=True)
        
        return scored_daydreams[:keep_top]
    
    def save_accepted_daydreams(self, accepted_daydreams: List[Dict[str, Any]], output_path: str):
        """Save accepted daydreams to JSONL file."""
        with open(output_path, 'w') as f:
            for daydream in accepted_daydreams:
                f.write(json.dumps(daydream) + '\n')
    
    def get_statistics(self, accepted_daydreams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the accepted daydreams."""
        if not accepted_daydreams:
            return {}
        
        scores = [d['critic'] for d in accepted_daydreams]
        
        stats = {
            'count': len(accepted_daydreams),
            'average_novelty': np.mean([s['novelty'] for s in scores]),
            'average_coherence': np.mean([s['coherence'] for s in scores]),
            'average_usefulness': np.mean([s['usefulness'] for s in scores]),
            'average_binding': np.mean([s['binding'] for s in scores]),
            'average_overall': np.mean([s['overall_score'] for s in scores]),
            'top_score': max(s['overall_score'] for s in scores),
            'score_distribution': np.histogram([s['overall_score'] for s in scores], bins=5)[0].tolist()
        }
        
        return stats


if __name__ == "__main__":
    # Test the critic
    critic = DDLCritic()
    
    # Mock daydream for testing
    test_daydream = {
        "hypothesis": "This hypothesis explores how sparse autoencoders could be combined with attention mechanisms to create more interpretable neural networks. By training sparse autoencoders on attention weight patterns, we might discover structured representations that reveal how models allocate computational resources.",
        "paper_anchor": "sparse autoencoder",
        "test": {
            "type": "ablation_control",
            "dataset_or_component": "transformer attention layers",
            "manipulation": "replace standard attention with sparse-autoencoder-guided attention",
            "metric": "interpretability score and task accuracy",
            "expected_direction": "increase",
            "success_threshold": 0.7,
            "timeframe_days": 21
        }
    }
    
    print("Testing critic evaluation...")
    scores = critic.evaluate_daydream(test_daydream)
    print(f"Valid: {scores.valid}")
    print(f"Scores: novelty={scores.novelty:.2f}, coherence={scores.coherence:.2f}, usefulness={scores.usefulness:.2f}, binding={scores.binding:.2f}")
    if not scores.valid:
        print(f"Reasons: {scores.reasons}")