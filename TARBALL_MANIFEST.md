# DDL V8 Hybrid System - Tarball Manifest

**File**: `DDL_V8_Hybrid_System_20250825_083724.tar.gz`  
**Size**: 228KB  
**Created**: August 25, 2025 08:37:24  
**Total Files**: 41

## Contents Overview

### Core Pipeline Scripts (8 files)
- `ddl_paper_only_pipeline.py` - Main orchestrator with hybrid model routing
- `ddl_binder.py` - Evidence binding (configured for Ollama qwen3-30b)
- `ddl_generator.py` - Daydream generation (uses LM Studio 235B)
- `ddl_critic.py` - Quality evaluation and filtering
- `semantic_ddl_pipeline.py` - Semantic concept pairing system
- `weave_composer_freeflow.py` - Final editorial composition
- `paper_concepts.py` - LLM-based paper concept extraction
- `hybrid_retrieval_system.py` - BGE-M3 + BM25 vector search

### Concept & Sampling Components (4 files)
- `semantic_corpus_sampler.py` - Intelligent corpus concept matching
- `ddl_sampler_ephemeral.py` - Ephemeral daydream sampling system
- `llm_concept_extractor.py` - LLM interface for concept extraction

### Configuration & Setup (4 files)
- `config.json` - Hybrid model configuration (235B + 30B setup)
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive V8 system documentation
- `V8_SETUP.md` - Technical setup and architecture notes

### Testing & Utilities (6 files)
- `test_v6_readiness.py` - System readiness validation
- `test_config_fixes.py` - Configuration testing
- `quick_config_test.py` - Quick configuration validation
- `quick_corpus_test.py` - Corpus connectivity testing
- `run_enhanced_4x_test.py` - Enhanced scaling test harness
- `run_v6_with_logging.py` - Logging-enabled test runner

### Input Papers (9 files)
Located in `pipeline/input/`:
- `CBH.txt` - Critical Brain Hypothesis paper
- `Autoalignment.txt` - Auto-alignment safety research
- `honest.txt` - Honesty in AI systems
- `backdoor.txt` - Backdoor attack analysis
- `Threat.txt` - AI threat modeling
- `Epistemic.txt` - Epistemic considerations
- `Prior.txt` - Prior knowledge integration
- `Monofact.txt` - Monofactual reasoning
- `problem.txt` - Problem specification

### Example Output (10 files)
Complete test run: `pipeline/output/DDL_Cbh_20250825_080551/`
- `ddl/concepts.jsonl` - Extracted paper concepts
- `ddl/pairs.sampled.jsonl` - Generated concept pairs
- `ddl/daydreams.generated.jsonl` - Raw hypotheses (3 daydreams)
- `ddl/daydreams.accepted.jsonl` - Filtered daydreams (2 accepted)
- `ddl/daydreams.bound.jsonl` - Evidence-bound daydreams (0 bound)
- `ddl/daydreams.paper_only.jsonl` - Paper-only evidence
- `ddl/daydreams.unbound.jsonl` - Unbound daydreams (2 unbound)
- `ddl/editorial_freeflow.md` - Final 2,167-word analysis
- `manifest.json` - Run metadata

## System Requirements

### Model Services
1. **LM Studio** (port 1234) - `unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF`
2. **Ollama** (port 11434) - `qwen3-30b`

### Dependencies
- Python 3.9+
- BGE-M3 embeddings model
- FAISS vector database
- Sentence transformers
- Pandas, numpy, openai client

### Corpus
- Shared corpus at: `/Users/willkirby/scrape 2/LW_scrape/chunked_corpus/contextual_chunks_complete.parquet`
- 66,716 contextual chunks from AI safety literature

## Quick Deployment

```bash
# Extract
tar -xzf DDL_V8_Hybrid_System_20250825_083724.tar.gz
cd extracted_directory

# Install dependencies
pip3 install -r requirements.txt

# Test systems
python3 quick_config_test.py

# Run sample
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 3
```

## Performance Metrics

Based on included example output (`DDL_Cbh_20250825_080551`):
- **Runtime**: ~26 minutes for 3-pair analysis
- **Generated**: 3 daydreams, 2 accepted
- **Evidence**: 0 bound (using fallback enhanced composition)
- **Output**: 2,167 words of CBH-focused analysis
- **Speed**: Hybrid system completed successfully

This tarball contains everything needed to deploy and run the DDL V8 Hybrid System with working examples and comprehensive documentation.