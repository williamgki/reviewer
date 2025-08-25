#!/usr/bin/env python3

import os
from pathlib import Path

def test_v6_system_readiness():
    """Test if v6 system has all required components for end-to-end run."""
    
    print("🧪 Testing v6 System Readiness...")
    print("=" * 50)
    
    issues = []
    
    # 1. Check core files
    core_files = [
        'ddl_paper_only_pipeline.py',
        'paper_concepts.py', 
        'llm_concept_extractor.py',
        'semantic_corpus_sampler.py',
        'semantic_ddl_pipeline.py',
        'hybrid_retrieval_system.py',
        'ddl_generator.py',
        'ddl_critic.py',
        'ddl_binder.py',
        'weave_composer_freeflow.py',
        'config.json'
    ]
    
    print("1. Core Files:")
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            issues.append(f"Missing core file: {file}")
    
    # 2. Check directories
    dirs = [
        'pipeline/input',
        'pipeline/output',
        '../retrieval_indexes',
        '../chunked_corpus'
    ]
    
    print("\n2. Required Directories:")
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            issues.append(f"Missing directory: {dir_path}")
    
    # 3. Check critical index files
    index_files = [
        '../retrieval_indexes/bge_m3_embeddings_mac.pkl',
        '../retrieval_indexes/bge_m3_faiss_mac.index'
    ]
    
    print("\n3. Vector Database Files:")
    for file in index_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            issues.append(f"Missing index file: {file}")
    
    # 4. Check test paper
    print("\n4. Test Input:")
    if os.path.exists('pipeline/input/honest.txt'):
        print("   ✅ pipeline/input/honest.txt")
    else:
        print("   ❌ pipeline/input/honest.txt")
        issues.append("Missing test paper")
    
    # 5. Test imports
    print("\n5. Import Tests:")
    try:
        from paper_concepts import PaperConceptExtractor
        print("   ✅ PaperConceptExtractor")
    except Exception as e:
        print(f"   ❌ PaperConceptExtractor: {e}")
        issues.append(f"Import error: PaperConceptExtractor - {e}")
    
    try:
        from hybrid_retrieval_system import HybridRetrievalSystem
        print("   ✅ HybridRetrievalSystem")
    except Exception as e:
        print(f"   ❌ HybridRetrievalSystem: {e}")
        issues.append(f"Import error: HybridRetrievalSystem - {e}")
    
    try:
        from semantic_ddl_pipeline import SemanticDDLPipeline
        print("   ✅ SemanticDDLPipeline")
    except Exception as e:
        print(f"   ❌ SemanticDDLPipeline: {e}")
        issues.append(f"Import error: SemanticDDLPipeline - {e}")
    
    # 6. Test LLM connection
    print("\n6. LLM Connection:")
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("   ✅ LLM server is running")
        else:
            print(f"   ❌ LLM server error: {response.status_code}")
            issues.append("LLM server not responding correctly")
    except Exception as e:
        print(f"   ❌ LLM server not accessible: {e}")
        issues.append(f"LLM server issue: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if issues:
        print(f"❌ Found {len(issues)} issues that need to be resolved:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n🔧 System needs fixes before end-to-end testing")
        return False
    else:
        print("✅ All systems ready! v6 should work end-to-end")
        return True

if __name__ == "__main__":
    ready = test_v6_system_readiness()
    if ready:
        print("\n🚀 Ready to run: python3 ddl_paper_only_pipeline.py pipeline/input/honest.txt --pairs 3")
    else:
        print("\n⚠️ Fix the issues above first, then test again")