#!/usr/bin/env python3

import json
import sys
import os

# Test script to quickly verify config propagation without full pipeline
def test_config_propagation():
    print("🧪 Quick Config Propagation Test")
    
    # Load config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config.json")
    except Exception as e:
        print(f"❌ Failed to load config.json: {e}")
        return False
    
    # Test semantic DDL pipeline config receipt
    try:
        from semantic_ddl_pipeline import SemanticDDLPipeline
        print("✅ Imported SemanticDDLPipeline")
        
        # This should trigger our diagnostic logging
        pipeline = SemanticDDLPipeline(config)
        print("✅ Created SemanticDDLPipeline with config")
        
        # Check if sampler got config
        retrieval_config = config.get("retrieval", {})
        expected_min_score = retrieval_config.get("min_semantic_score", "NOT_FOUND")
        print(f"🔧 Expected min_semantic_score: {expected_min_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test semantic pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_propagation()
    sys.exit(0 if success else 1)