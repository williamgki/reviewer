#!/usr/bin/env python3

import json
import re
import openai
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class WeaveComposerFreeflow:
    """
    Free-flow narrative composer that creates thematic essays from daydream findings.
    Replaces rigid claim-card structure with flowing editorial narrative.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio", 
                 model: str = "gpt-oss-120b",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Composition parameters
        self.target_word_count = (2200, 3000)
        self.max_themes = 5
        self.min_themes = 3
        self.max_quotes_total = 30
        self.max_findings_per_theme = 6
        self.min_findings_per_theme = 3
    
    def compose_review(self, bound_daydreams: List[Dict[str, Any]], 
                      paper_title: str, paper_text: str, paper_only_leads: List[Dict[str, Any]] = None) -> str:
        """
        Compose a free-flow review from bound daydreams and paper-only leads.
        
        Args:
            bound_daydreams: Daydreams with evidence bound
            paper_title: Title of the paper being reviewed
            paper_text: Full paper text for context
            paper_only_leads: Paper-only daydreams with Boolean queries for search plan
            
        Returns:
            Complete markdown review
        """
        paper_only_leads = paper_only_leads or []
        print(f"Composing free-flow review from {len(bound_daydreams)} daydreams...")
        
        # 1. Cluster daydreams into themes
        themes = self._cluster_daydreams_into_themes(bound_daydreams)
        print(f"Identified {len(themes)} themes")
        
        # 2. Generate lead section
        lead = self._generate_lead_section(paper_title, paper_text, themes)
        
        # 3. Compose theme sections
        theme_sections = []
        total_quotes_used = 0
        
        for theme_name, theme_daydreams in themes.items():
            if total_quotes_used >= self.max_quotes_total:
                break
            
            section = self._compose_theme_section(theme_name, theme_daydreams, 
                                                self.max_quotes_total - total_quotes_used)
            if section:
                theme_sections.append(section)
                # Count quotes used in this section
                quotes_in_section = len(re.findall(r'\[(\d+)\]', section))
                total_quotes_used += quotes_in_section
        
        # 4. Generate relevant work section
        relevant_work = self._generate_relevant_work_section(bound_daydreams)
        
        # 5. Generate editorial take
        editorial_take = self._generate_editorial_take(themes, paper_title)
        
        # 6. Generate margin notes
        margin_notes = self._generate_margin_notes(bound_daydreams)
        
        # 7. Build reference list
        references = self._build_reference_list(bound_daydreams)
        
        # 8. Assemble final review
        review = self._assemble_final_review(
            lead, theme_sections, relevant_work, editorial_take, 
            margin_notes, references, paper_title
        )
        
        # 9. Validate and adjust
        word_count = len(review.split())
        print(f"Generated review: {word_count} words, {total_quotes_used} quotes")
        
        return review
    
    def _cluster_daydreams_into_themes(self, bound_daydreams: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster daydreams into thematic groups using embeddings and TF-IDF naming."""
        if len(bound_daydreams) <= self.min_themes:
            themes = {}
            for i, daydream in enumerate(bound_daydreams):
                theme_name = daydream.get('theme_hint', f'Theme {i+1}')
                themes[theme_name] = [daydream]
            return themes

        try:
            # Build embedding texts
            texts = []
            for d in bound_daydreams:
                pieces = [
                    d.get('hypothesis', ''),
                    d.get('paper_concept', ''),
                    d.get('corpus_concept', ''),
                    d.get('theme_hint', ''),
                    d.get('paper_anchor', ''),
                ]
                for ev in d.get('evidence', []):
                    pieces.append(ev.get('quote', ''))
                    pieces.append(ev.get('section_title', ''))
                texts.append(" ".join([p for p in pieces if p]))

            embeddings = self.embedding_model.encode(texts)

            possible_k = range(self.min_themes, min(self.max_themes, len(bound_daydreams)) + 1)
            best_score, best_labels = -1, None
            for k in possible_k:
                if k <= 1 or k >= len(bound_daydreams):
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score, best_labels = score, labels

            if best_labels is None:
                raise ValueError("Clustering failed")

            clusters = defaultdict(list)
            for label, daydream in zip(best_labels, bound_daydreams):
                clusters[label].append(daydream)
        except Exception:
            # Fallback to theme hints if clustering fails
            clusters = defaultdict(list)
            for d in bound_daydreams:
                clusters[d.get('theme_hint', 'General')].append(d)

        themes = {}
        for idx, daydreams in clusters.items():
            # Create cluster name using TF-IDF
            corpus = []
            for d in daydreams:
                corpus.append(d.get('hypothesis', ''))
                corpus.append(d.get('paper_concept', ''))
                corpus.append(d.get('corpus_concept', ''))
                corpus.append(d.get('theme_hint', ''))
                corpus.append(d.get('paper_anchor', ''))
                for ev in d.get('evidence', []):
                    corpus.append(ev.get('quote', ''))
                    corpus.append(ev.get('section_title', ''))
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf = vectorizer.fit_transform(corpus)
                mean_scores = np.asarray(tfidf.mean(axis=0)).ravel()
                feature_names = vectorizer.get_feature_names_out()
                top_indices = mean_scores.argsort()[::-1][:3]
                top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
                theme_name = " ".join(top_terms).title() if top_terms else f"Theme {idx+1}"
            except Exception:
                theme_name = f"Theme {idx+1}"

            # Rank daydreams inside theme by critic score and cap findings
            ranked_daydreams = sorted(
                daydreams,
                key=lambda d: d.get('critic', {}).get('overall_score', 0),
                reverse=True
            )[: self.max_findings_per_theme]

            themes[theme_name] = ranked_daydreams

        def theme_metrics(daydreams: List[Dict[str, Any]]) -> Tuple[float, float, float]:
            sources: Set[str] = set()
            scores, novelties = [], []
            for d in daydreams:
                scores.append(d.get('critic', {}).get('overall_score', 0))
                novelties.append(d.get('critic', {}).get('novelty', 0))
                for ev in d.get('evidence', []):
                    src = ev.get('citation') or ev.get('source')
                    if src:
                        sources.add(src)
            breadth = len(sources)
            return (
                breadth,
                float(np.mean(scores)) if scores else 0.0,
                float(np.mean(novelties)) if novelties else 0.0,
            )

        # Rank themes by source breadth, critic score, novelty
        ordered = sorted(
            themes.items(),
            key=lambda item: theme_metrics(item[1]),
            reverse=True,
        )[: self.max_themes]

        return {name: daydreams for name, daydreams in ordered}
    
    def _generate_lead_section(self, paper_title: str, paper_text: str, 
                              themes: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate lead section introducing the paper and review approach."""
        
        # Extract paper abstract or first few sentences for context
        paper_context = self._extract_paper_context(paper_text)
        theme_names = list(themes.keys())
        
        system_prompt = """Write a lead section (200-300 words) for an academic review that introduces:

1. What the paper argues and its main contributions
2. Where it sits in the broader research landscape  
3. What this review explores (daydreaming paper concepts against existing literature)

Style: Engaging but scholarly, plain language, no boilerplate phrases.
Structure: 2-3 short paragraphs, average sentence length ≤18 words.
Focus: Set up the thematic exploration that follows."""
        
        user_prompt = f"""PAPER TITLE: {paper_title}

PAPER CONTEXT: {paper_context}

THEMES TO EXPLORE: {', '.join(theme_names)}

Write the lead section introducing this paper and the thematic analysis approach."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Failed to generate lead section: {e}")
            return f"# Review of {paper_title}\n\nThis review explores the paper's core concepts against existing literature through thematic analysis."
    
    def _extract_paper_context(self, paper_text: str, max_length: int = 500) -> str:
        """Extract key context from paper (abstract or opening)."""
        # Try to find abstract
        abstract_match = re.search(r'abstract[:\s]*(.*?)(?:\n\n|\nintroduction)', 
                                  paper_text.lower(), re.DOTALL)
        
        if abstract_match:
            return abstract_match.group(1).strip()[:max_length]
        
        # Fallback to first few sentences
        sentences = re.split(r'[.!?]+', paper_text)
        context = ""
        for sentence in sentences[:5]:
            if len(context + sentence) < max_length:
                context += sentence.strip() + ". "
            else:
                break
        
        return context.strip()
    
    def _compose_theme_section(self, theme_name: str, theme_daydreams: List[Dict[str, Any]], 
                              quote_budget: int) -> str:
        """Compose a thematic section from daydream findings."""
        
        # Prepare findings data
        findings_data = []
        for daydream in theme_daydreams[:self.max_findings_per_theme]:
            finding = {
                'hypothesis': daydream['hypothesis'],
                'test_summary': self._summarize_test(daydream['test']),
                'evidence': daydream['evidence'],
                'concepts': [daydream['paper_concept'], daydream['corpus_concept']]
            }
            findings_data.append(finding)
        
        system_prompt = """Write a thematic section for an academic review that:

1. Weaves together 3-6 Daydream Findings naturally
2. Includes 1-2 quotes per finding as inline citations [X]  
3. Contrasts supports vs contradicts evidence
4. Calls out what the original paper might be missing
5. Summarizes the 2 strongest tests (1 line each)

Style: Plain language, flowing narrative, short paragraphs (≤5 sentences).
Evidence: Only assert facts supported by provided quotes.
Structure: Theme introduction → findings integration → test summary → gaps identified.

Limit total quotes to fit budget. Use footnote-style citations [1], [2], etc."""
        
        user_prompt = f"""THEME: {theme_name}

FINDINGS: {json.dumps(findings_data, indent=2)}

QUOTE BUDGET: {quote_budget} remaining

Write the thematic section integrating these findings."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=800,
                timeout=45
            )
            
            section = response.choices[0].message.content.strip()
            
            # Add section header
            return f"## {theme_name}\n\n{section}\n"
            
        except Exception as e:
            print(f"Failed to generate theme section {theme_name}: {e}")
            return f"## {theme_name}\n\n[Section generation failed]\n"
    
    def _summarize_test(self, test: Dict[str, Any]) -> str:
        """Create a brief test summary."""
        test_type = test.get('type', 'unknown')
        manipulation = test.get('manipulation', '')
        metric = test.get('metric', '')
        direction = test.get('expected_direction', '')
        
        return f"{test_type}: {manipulation} → measure {metric} (expect {direction})"
    
    def _generate_relevant_work_section(self, bound_daydreams: List[Dict[str, Any]]) -> str:
        """Generate relevant work section from corpus citations."""
        
        # Extract unique authors and works
        corpus_works = set()
        for daydream in bound_daydreams:
            for evidence in daydream['evidence']:
                if evidence['source'] == 'corpus':
                    citation = evidence['citation']
                    # Extract author and year
                    match = re.search(r'\(([^,]+),\s*(\d+)', citation)
                    if match:
                        author, year = match.groups()
                        corpus_works.add((author, year, citation))
        
        if not corpus_works:
            return ""
        
        # Format as bullet list
        work_list = []
        for author, year, citation in sorted(corpus_works):
            # Create one-liner description
            work_list.append(f"• {citation}: [Brief description of relevance]")
        
        relevant_work = "## Relevant Work Surfaced\n\n" + "\n".join(work_list[:12])  # Limit to 12 references
        
        return relevant_work + "\n"
    
    def _generate_editorial_take(self, themes: Dict[str, List[Dict[str, Any]]], 
                               paper_title: str) -> str:
        """Generate editorial conclusion."""
        
        theme_summaries = []
        for theme_name, daydreams in themes.items():
            avg_score = np.mean([d['critic']['overall_score'] for d in daydreams])
            theme_summaries.append(f"{theme_name} (strength: {avg_score:.2f})")
        
        system_prompt = """Write an editorial take (150-250 words) that:

1. Synthesizes what would change your mind about the paper's claims
2. Recommends concrete next steps for researchers
3. Highlights the most promising research directions uncovered

Style: Thoughtful, balanced, actionable.
Structure: 2-3 short paragraphs.
Tone: What a senior researcher would think after deep analysis."""
        
        user_prompt = f"""PAPER: {paper_title}

THEMES ANALYZED: {theme_summaries}

Write editorial conclusion with recommended next steps."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=350,
                timeout=25
            )
            
            content = response.choices[0].message.content.strip()
            return f"## Editorial Take\n\n{content}\n"
            
        except Exception as e:
            print(f"Failed to generate editorial take: {e}")
            return "## Editorial Take\n\n[Editorial generation failed]\n"
    
    def _generate_margin_notes(self, bound_daydreams: List[Dict[str, Any]]) -> str:
        """Generate crisp, paste-able margin notes."""
        
        # Extract key insights for margin notes
        insights = []
        for daydream in bound_daydreams[:10]:  # Limit to top daydreams
            concept_pair = f"{daydream['paper_concept']} × {daydream['corpus_concept']}"
            score = daydream['critic']['overall_score']
            insights.append((concept_pair, score))
        
        # Sort by score and format
        insights.sort(key=lambda x: x[1], reverse=True)
        
        margin_notes = "## Margin Notes\n\n"
        for i, (insight, score) in enumerate(insights[:8]):
            margin_notes += f"{i+1}. {insight} (score: {score:.2f})\n"
        
        return margin_notes + "\n"
    
    def _build_reference_list(self, bound_daydreams: List[Dict[str, Any]]) -> str:
        """Build numbered reference list from citations."""
        
        # Extract all unique citations
        all_citations = set()
        for daydream in bound_daydreams:
            for evidence in daydream['evidence']:
                citation = evidence['citation']
                if citation != '(Authors, 2024; paper)':  # Skip paper self-references
                    all_citations.add(citation)
        
        # Number citations
        references = "## References\n\n"
        for i, citation in enumerate(sorted(all_citations), 1):
            references += f"[{i}] {citation}\n"
        
        return references + "\n"
    
    def _assemble_final_review(self, lead: str, theme_sections: List[str], 
                              relevant_work: str, editorial_take: str, 
                              margin_notes: str, references: str,
                              paper_title: str) -> str:
        """Assemble all sections into final review."""
        
        review_parts = [
            f"# Deep Dive Review: {paper_title}\n",
            f"*Generated by DDL Paper-Only Mode*\n\n---\n",
            lead,
            "\n---\n",
            "\n".join(theme_sections),
            relevant_work,
            editorial_take,
            margin_notes,
            references
        ]
        
        return "\n".join(review_parts)
    
    def save_review(self, review_content: str, output_path: str):
        """Save the composed review to file."""
        with open(output_path, 'w') as f:
            f.write(review_content)
    
    def get_review_statistics(self, review_content: str) -> Dict[str, Any]:
        """Get statistics about the generated review."""
        word_count = len(review_content.split())
        quote_count = len(re.findall(r'\[(\d+)\]', review_content))
        theme_count = len(re.findall(r'^##\s+(?!References|Editorial|Margin|Relevant)', review_content, re.MULTILINE))
        
        # Count unique authors mentioned
        author_matches = re.findall(r'\(([^,)]+),\s*\d+', review_content)
        unique_authors = len(set(author_matches))
        
        return {
            'word_count': word_count,
            'quote_count': quote_count, 
            'theme_count': theme_count,
            'unique_authors': unique_authors,
            'within_target': self.target_word_count[0] <= word_count <= self.target_word_count[1]
        }


if __name__ == "__main__":
    # Test the composer
    composer = WeaveComposerFreeflow()
    
    # Mock bound daydream for testing
    test_daydream = {
        'paper_concept': 'sparse autoencoder',
        'corpus_concept': 'attention mechanism', 
        'hypothesis': 'Sparse autoencoders could enhance attention interpretability...',
        'test': {
            'type': 'ablation_control',
            'manipulation': 'replace attention with sparse-guided attention',
            'metric': 'interpretability score',
            'expected_direction': 'increase'
        },
        'critic': {'overall_score': 0.75},
        'evidence': [
            {
                'source': 'paper',
                'label': 'supports',
                'citation': '(Authors, 2024; paper)',
                'quote': 'Sparse autoencoders learn interpretable features...'
            }
        ],
        'theme_hint': 'interpretability & attention'
    }
    
    print("Testing theme clustering...")
    themes = composer._cluster_daydreams_into_themes([test_daydream])
    print(f"Generated {len(themes)} themes: {list(themes.keys())}")