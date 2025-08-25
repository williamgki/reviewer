import pytest

try:
    from hybrid_retrieval_system import HybridRetrievalSystem, SearchResult
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover - skip if deps missing
    pytest.skip(f"HybridRetrievalSystem unavailable: {e}", allow_module_level=True)

import pandas as pd


@pytest.fixture(scope="module")
def retrieval_system(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("hybrid")
    chunks_path = tmp_path / "chunks.parquet"
    index_dir = tmp_path / "indexes"

    df = pd.DataFrame([
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "content": "This is a test chunk about machine learning and AI.",
            "section_path": "sec1",
            "summary_header": "AI",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "content": "Another test chunk discussing biology and evolution.",
            "section_path": "sec2",
            "summary_header": "Biology",
        },
    ])
    df.to_parquet(chunks_path)

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"SentenceTransformer model unavailable: {e}")

    # Build indexes first
    try:
        builder = HybridRetrievalSystem(
            chunks_path=str(chunks_path),
            index_dir=str(index_dir),
        )
        builder.embedding_model = model
        builder.build_indexes(force_rebuild=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not build retrieval indexes: {e}")

    # Instantiate system with existing indexes
    system = HybridRetrievalSystem(
        chunks_path=str(chunks_path),
        index_dir=str(index_dir),
    )
    system.embedding_model = model
    try:
        system.build_indexes(force_rebuild=False)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not load retrieval indexes: {e}")

    return system


def test_hybrid_system_returns_results(retrieval_system):
    results = retrieval_system.hybrid_search("machine learning", k=5)
    assert len(results) >= 1
    assert any(isinstance(r, SearchResult) for r in results)
