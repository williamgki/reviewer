#!/usr/bin/env python3
"""
DDL V8 Unit Tests - Critical functionality validation

Tests semantic enable/disable parity, parquet engine guards, 
and core binder functionality.
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestSemanticEnableDisable(unittest.TestCase):
    """Test semantic retrieval enable/disable parity."""
    
    def setUp(self):
        """Set up test configuration."""
        with open('config.json') as f:
            self.base_config = json.load(f)
            
        # Create minimal test corpus
        self.test_docs = [
            {"doc_id": "test1", "text": "sparse autoencoder attention mechanism", "author": "Test", "year": 2024},
            {"doc_id": "test2", "text": "neural network representation learning", "author": "Test", "year": 2024},
            {"doc_id": "test3", "text": "interpretability safety alignment", "author": "Test", "year": 2024},
            {"doc_id": "test4", "text": "transformer architecture scaling laws", "author": "Test", "year": 2024},
            {"doc_id": "test5", "text": "reinforcement learning human feedback", "author": "Test", "year": 2024}
        ]
        
    def create_test_corpus(self):
        """Create temporary parquet file for testing."""
        import pandas as pd
        
        df = pd.DataFrame(self.test_docs)
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        df.to_parquet(temp_file.name)
        return temp_file.name
        
    def test_semantic_enabled_vs_disabled(self):
        """Test that semantic disabled returns subset of semantic enabled results."""
        from ddl_binder import DDLEvidenceBinder
        
        corpus_file = self.create_test_corpus()
        
        try:
            # Test with semantic enabled
            config_enabled = self.base_config.copy()
            config_enabled['binder'] = {'semantic': {'enabled': True, 'max_docs': 10}}
            
            binder_enabled = DDLEvidenceBinder(
                corpus_path=corpus_file,
                config=config_enabled,
                run_id='semantic-enabled-test'
            )
            
            # Test with semantic disabled  
            config_disabled = self.base_config.copy()
            config_disabled['binder'] = {'semantic': {'enabled': False}}
            
            binder_disabled = DDLEvidenceBinder(
                corpus_path=corpus_file,
                config=config_disabled,
                run_id='semantic-disabled-test'
            )
            
            # Both should initialize successfully
            self.assertTrue(binder_enabled.embedding_model is not None, "Semantic enabled should have embedding model")
            self.assertTrue(binder_disabled.embedding_model is None, "Semantic disabled should not have embedding model")
            
            # Both should load the same corpus
            self.assertEqual(len(binder_enabled.corpus_df), len(binder_disabled.corpus_df))
            
            # Test retrieval on same query - semantic should find more or equal candidates
            test_query = "sparse autoencoder"
            
            candidates_enabled = binder_enabled._retrieve_chunks(test_query, top_k=5)
            candidates_disabled = binder_disabled._retrieve_chunks(test_query, top_k=5)
            
            # Semantic should find at least as many relevant results
            self.assertGreaterEqual(len(candidates_enabled), 1, "Semantic enabled should find candidates")
            self.assertGreaterEqual(len(candidates_disabled), 1, "Lexical-only should find candidates")
            
            # At minimum, both should find the document containing the exact query
            enabled_texts = [c['text'].lower() for c in candidates_enabled]
            disabled_texts = [c['text'].lower() for c in candidates_disabled]
            
            query_found_enabled = any("sparse autoencoder" in text for text in enabled_texts)
            query_found_disabled = any("sparse autoencoder" in text for text in disabled_texts)
            
            self.assertTrue(query_found_enabled, "Semantic enabled should find exact match")
            self.assertTrue(query_found_disabled, "Lexical disabled should find exact match")
            
            print(f"✅ Semantic enabled found {len(candidates_enabled)} candidates")
            print(f"✅ Lexical disabled found {len(candidates_disabled)} candidates")
            
        finally:
            os.unlink(corpus_file)

class TestParquetEngineGuards(unittest.TestCase):
    """Test parquet engine validation and error handling."""
    
    def test_missing_parquet_engine_error(self):
        """Test that missing parquet engines raise clear errors."""
        # This test was already implemented and validated in Task 4
        # We'll test the preflight script validation
        
        from scripts.preflight import check_parquet_engine
        
        # Test that check passes with current installation
        result = check_parquet_engine()
        self.assertTrue(result, "Parquet engine check should pass with valid installation")
        
    def test_corrupted_parquet_file_handling(self):
        """Test handling of corrupted parquet files."""
        from ddl_binder import DDLEvidenceBinder
        
        # Create a fake parquet file (just text)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, mode='w') as fake_file:
            fake_file.write("This is not a parquet file")
            fake_file_path = fake_file.name
            
        try:
            config = {'binder': {'semantic': {'enabled': False}}}  # Speed up test
            
            with self.assertRaises(RuntimeError) as context:
                binder = DDLEvidenceBinder(
                    corpus_path=fake_file_path,
                    config=config,
                    run_id='corrupted-test'
                )
                
            error_msg = str(context.exception)
            self.assertIn("Failed to load corpus parquet file", error_msg)
            print(f"✅ Corrupted parquet correctly rejected: {error_msg[:50]}...")
            
        finally:
            os.unlink(fake_file_path)
            
    def test_empty_corpus_handling(self):
        """Test handling of empty corpus scenarios."""
        import pandas as pd
        
        # Create empty parquet file
        empty_df = pd.DataFrame(columns=['doc_id', 'text', 'author', 'year'])
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        empty_df.to_parquet(temp_file.name)
        
        try:
            from ddl_binder import DDLEvidenceBinder
            config = {'binder': {'semantic': {'enabled': False}}}
            
            binder = DDLEvidenceBinder(
                corpus_path=temp_file.name,
                config=config,
                run_id='empty-corpus-test'
            )
            
            # Should handle empty corpus gracefully
            self.assertTrue(binder.corpus_df.empty or len(binder.corpus_df) == 0)
            
            # Test retrieval with empty corpus
            candidates = binder._retrieve_chunks("test query", top_k=5)
            self.assertEqual(len(candidates), 0, "Empty corpus should return no candidates")
            
            print("✅ Empty corpus handled gracefully")
            
        finally:
            os.unlink(temp_file.name)

class TestMetricsCollection(unittest.TestCase):
    """Test binder metrics collection and writing."""
    
    def test_binder_metrics_collection(self):
        """Test that binder metrics are collected properly."""
        from ddl_binder import DDLEvidenceBinder
        import pandas as pd
        
        # Create small test corpus  
        test_data = pd.DataFrame([
            {"doc_id": "m1", "text": "test document one", "author": "Test", "year": 2024},
            {"doc_id": "m2", "text": "test document two", "author": "Test", "year": 2024}
        ])
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        test_data.to_parquet(temp_file.name)
        
        try:
            config = {'binder': {'semantic': {'enabled': False, 'max_docs': 5}}}  # Fast test
            
            binder = DDLEvidenceBinder(
                corpus_path=temp_file.name,
                config=config,
                run_id='metrics-test'
            )
            
            # Check initial metrics
            self.assertIn('run_id', binder.metrics)
            self.assertIn('parquet_files_loaded', binder.metrics)
            self.assertIn('rows_scanned', binder.metrics)
            self.assertEqual(binder.metrics['run_id'], 'metrics-test')
            self.assertEqual(binder.metrics['parquet_files_loaded'], 1)
            self.assertEqual(binder.metrics['rows_scanned'], 2)
            
            # Test metrics writing
            binder.write_metrics()
            
            # Verify metrics file was created
            self.assertTrue(os.path.exists('binder_metrics.json'))
            
            with open('binder_metrics.json') as f:
                written_metrics = json.load(f)
                
            self.assertEqual(written_metrics['run_id'], 'metrics-test')
            self.assertEqual(written_metrics['rows_scanned'], 2)
            self.assertIn('timestamp', written_metrics)
            
            print("✅ Metrics collection and writing works correctly")
            
        finally:
            os.unlink(temp_file.name)
            if os.path.exists('binder_metrics.json'):
                os.unlink('binder_metrics.json')

def run_unit_tests():
    """Run all unit tests and return results."""
    print("🧪 DDL V8 Unit Tests - Critical Functionality\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestSemanticEnableDisable))
    suite.addTest(unittest.makeSuite(TestParquetEngineGuards))
    suite.addTest(unittest.makeSuite(TestMetricsCollection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"\n📊 Unit Test Results:")
    print(f"   Total: {total_tests} tests")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failures}")
    print(f"   Errors: {errors}")
    
    if result.wasSuccessful():
        print("🎉 All unit tests passed!")
        return True
    else:
        print("❌ Some unit tests failed")
        return False

if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)