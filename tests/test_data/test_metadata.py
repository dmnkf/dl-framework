import numpy as np
import pytest
import torch
import threading
import concurrent.futures
import logging
from pathlib import Path
from queue import Queue
from datetime import datetime

from src.data.metadata_store import (
    JSONMetadataStoreBackend,
    SQLiteMetadataStoreBackend,
    MetadataStore
)
from tests.helpers.mock_ecg_data import create_mock_metadata


logger = logging.getLogger(__name__)


@pytest.fixture
def temp_json_store(tmp_path):
    """Fixture for JSON metadata store."""
    return JSONMetadataStoreBackend(tmp_path)


@pytest.fixture
def temp_sqlite_store(tmp_path):
    """Fixture for SQLite metadata store."""
    return SQLiteMetadataStoreBackend(tmp_path)


@pytest.fixture
def mock_metadata():
    """Fixture for mock metadata."""
    return create_mock_metadata(n_samples=10)


def test_json_metadata_store_basic(temp_json_store, mock_metadata):
    """Test basic operations of JSON metadata store."""
    dataset_key = "test_dataset"

    # Test storing
    temp_json_store.set(mock_metadata)

    # Test retrieval
    for record_id in mock_metadata:
        stored = temp_json_store.get(record_id)
        assert stored == mock_metadata[record_id]


def test_sqlite_metadata_store_basic(temp_sqlite_store, mock_metadata):
    """Test basic operations of SQLite metadata store."""
    # Test storing
    temp_sqlite_store.set(mock_metadata)

    # Test retrieval
    for record_id in mock_metadata:
        stored = temp_sqlite_store.get(record_id)
        assert stored == mock_metadata[record_id]


@pytest.mark.parametrize("backend", ["json"])
def test_metadata_store_batch_operations(backend, tmp_path, mock_metadata):
    """Test batch operations for both backend types."""
    store = MetadataStore(data_root=tmp_path)

    # Store metadata
    store.set(mock_metadata)

    # Test batch retrieval
    record_ids = list(mock_metadata.keys())[:5]
    batch_data = store.get_batch(record_ids)

    assert len(batch_data) == len(record_ids)
    for record_id in record_ids:
        assert batch_data[record_id] == mock_metadata[record_id]


def test_sqlite_batch_operations(temp_sqlite_store, mock_metadata):
    """Test batch operations specifically for SQLite backend."""
    # Store metadata
    temp_sqlite_store.set(mock_metadata)

    # Test batch retrieval
    record_ids = list(mock_metadata.keys())[:5]
    batch_data = temp_sqlite_store.get_batch(record_ids)

    assert len(batch_data) == len(record_ids)
    for record_id in record_ids:
        assert batch_data[record_id] == mock_metadata[record_id]


def test_metadata_store_tensor_conversion(tmp_path):
    """Test storing and retrieving metadata with tensor values."""
    store = MetadataStore(data_root=tmp_path)
    
    # Create metadata with tensors
    metadata = {
        "sample_1": {
            "features": torch.randn(10),
            "matrix": torch.randn(3, 3),
            "normal_value": 42,
        }
    }
    
    # Store metadata
    store.set(metadata)
            
    # Retrieve and verify
    retrieved = store.get("sample_1")
    assert "features" in retrieved
    assert "matrix" in retrieved
    assert "normal_value" in retrieved
    assert isinstance(retrieved["features"], list)
    assert isinstance(retrieved["matrix"], list)
    assert retrieved["normal_value"] == 42


def test_metadata_store_update_behavior(tmp_path):
    """Test metadata update behavior with overwrite flag."""
    store = MetadataStore(data_root=tmp_path)

    # Initial metadata
    initial_metadata = {"sample_1": {"value_1": 1, "value_2": 2}}
    store.set(initial_metadata)
    
    # Update with overwrite=False (default)
    update_metadata = {"sample_1": {"value_2": 3, "value_3": 4}}
    store.set(update_metadata, overwrite=False)

    # Check merged result
    merged = store.get("sample_1")
    assert merged["value_1"] == 1  # Original value preserved
    assert merged["value_2"] == 3  # Updated value
    assert merged["value_3"] == 4  # New value added

    # Update with overwrite=True
    store.set(update_metadata, overwrite=True)
    overwritten = store.get("sample_1")
    assert "value_1" not in overwritten  # Original value removed
    assert overwritten["value_2"] == 3
    assert overwritten["value_3"] == 4


def test_metadata_store_error_handling(tmp_path):
    """Test error handling in metadata store."""
    store = MetadataStore(data_root=tmp_path)
    
    # Test non-existent record
    empty_result = store.get("nonexistent")
    assert empty_result == {}
    
    # Test batch retrieval with non-existent records
    batch_result = store.get_batch(["nonexistent1", "nonexistent2"])
    assert all(not v for v in batch_result.values())


@pytest.mark.parametrize("backend", ["json"])
def test_metadata_store_large_data(backend, tmp_path):
    """Test handling of large metadata entries."""
    store = MetadataStore(data_root=tmp_path)

    # Create large metadata
    large_metadata = create_mock_metadata(n_samples=1000)
    
    # Add some large arrays
    for record_id in list(large_metadata.keys())[:10]:
        large_metadata[record_id]["large_array"] = np.random.randn(1000).tolist()

    # Test storing and retrieving
    store.set(large_metadata)

    # Verify a few random samples
    for record_id in list(large_metadata.keys())[:10]:
        retrieved = store.get(record_id)
        assert retrieved == large_metadata[record_id]
        assert len(retrieved["large_array"]) == 1000


def test_metadata_store_concurrent_access(tmp_path):
    """Test concurrent access patterns (basic simulation)."""
    store = MetadataStore(data_root=tmp_path)
    
    # Simulate concurrent writes
    metadata_1 = {"sample_1": {"value": 1}}
    metadata_2 = {"sample_1": {"other_value": 2}}
    
    store.set(metadata_1)
    store.set(metadata_2, overwrite=False)
    
    # Verify both values are present
    result = store.get("sample_1")
    assert result["value"] == 1
    assert result["other_value"] == 2


# New SQLite-specific tests
def test_sqlite_store_concurrent_access(tmp_path, caplog):
    """Test concurrent access patterns for SQLite backend."""
    caplog.set_level(logging.INFO)
    logger.info("Starting concurrent access test")
    
    store = SQLiteMetadataStoreBackend(tmp_path)
    n_threads = 4
    n_operations = 50
    error_queue = Queue()
    
    def worker(worker_id):
        thread_name = f"Worker-{worker_id}"
        thread = threading.current_thread()
        logger.info(f"Thread {thread_name} (id: {thread.ident}) started")
        
        try:
            for i in range(n_operations):
                start_time = datetime.now()
                
                # Write operation
                metadata = {f"sample_{worker_id}_{i}": {"value": i}}
                store.set(metadata)
                
                # Read operation
                result = store.get(f"sample_{worker_id}_{i}")
                if result["value"] != i:
                    raise AssertionError(f"Thread {thread_name}: Value mismatch. Expected {i}, got {result['value']}")
                
                duration = (datetime.now() - start_time).total_seconds()
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Thread {thread_name}: Operation {i} took {duration:.2f}s")
                
                if i % 10 == 0:  # Log progress every 10 operations
                    logger.info(f"Thread {thread_name}: Completed {i}/{n_operations} operations")
            
            logger.info(f"Thread {thread_name} completed successfully")
            
        except Exception as e:
            error_msg = f"Thread {thread_name} failed: {str(e)}"
            logger.error(error_msg)
            error_queue.put((thread_name, error_msg))
            raise
    
    logger.info(f"Starting {n_threads} threads with {n_operations} operations each")
    
    # Run concurrent operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(n_threads)]
        
        # Wait for completion and check for errors
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker failed: {e}")
                raise
    
    # Check for any errors in the queue
    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            thread_name, error_msg = error_queue.get()
            errors.append(f"{thread_name}: {error_msg}")
        raise AssertionError(f"Thread errors occurred:\n" + "\n".join(errors))
    
    # Verify final state
    all_metadata = store.get_all_metadata()
    expected_count = n_threads * n_operations
    actual_count = len(all_metadata)
    
    if actual_count != expected_count:
        logger.error(f"Data count mismatch. Expected {expected_count}, got {actual_count}")
        # Log the first few missing or extra items
        expected_keys = {f"sample_{w}_{i}" 
                        for w in range(n_threads) 
                        for i in range(n_operations)}
        actual_keys = set(all_metadata.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            logger.error(f"First 5 missing keys: {list(missing)[:5]}")
        if extra:
            logger.error(f"First 5 extra keys: {list(extra)[:5]}")
        
    assert actual_count == expected_count, f"Expected {expected_count} records, got {actual_count}"
    
    logger.info("Concurrent access test completed successfully")


def test_sqlite_store_transaction_rollback(tmp_path):
    """Test SQLite transaction rollback on error."""
    store = SQLiteMetadataStoreBackend(tmp_path)
    
    # Store initial data
    initial_data = {"sample_1": {"value": 1}}
    store.set(initial_data)
    
    # Attempt to store invalid data that should trigger rollback
    invalid_data = {"sample_2": {"value": torch.tensor([1, 2, 3])}}  # Non-serializable
    
    with pytest.raises(Exception):
        store.set(invalid_data)
    
    # Verify initial data is preserved
    assert store.get("sample_1") == {"value": 1}
    assert store.get("sample_2") == {}


def test_sqlite_store_connection_handling(tmp_path):
    """Test SQLite connection handling and cleanup."""
    # Test 1: Basic store operations
    store = SQLiteMetadataStoreBackend(tmp_path)
    store.set({"test": {"value": 1}})
    assert store.get("test") == {"value": 1}
    
    # Test 2: Data persists after store deletion
    db_path = store.db_path
    del store
    
    # Test 3: Data is accessible from a new store instance
    new_store = SQLiteMetadataStoreBackend(tmp_path)
    assert new_store.get("test") == {"value": 1}
    
    # Test 4: Multiple store instances work correctly
    store1 = SQLiteMetadataStoreBackend(tmp_path)
    store2 = SQLiteMetadataStoreBackend(tmp_path)
    
    # Both can read existing data
    assert store1.get("test") == {"value": 1}
    assert store2.get("test") == {"value": 1}
    
    # Both can write and read new data
    store1.set({"test2": {"value": 2}})
    assert store2.get("test2") == {"value": 2}
    
    store2.set({"test3": {"value": 3}})
    assert store1.get("test3") == {"value": 3}
    
    # Test 5: Clean shutdown doesn't affect data persistence
    del store1
    del store2
    
    final_store = SQLiteMetadataStoreBackend(tmp_path)
    assert final_store.get("test") == {"value": 1}
    assert final_store.get("test2") == {"value": 2}
    assert final_store.get("test3") == {"value": 3}


def test_sqlite_store_large_batch_operations(temp_sqlite_store):
    """Test handling of large batch operations in SQLite."""
    large_metadata = create_mock_metadata(n_samples=1000)
    
    # Test large batch write
    temp_sqlite_store.set(large_metadata)
    
    # Test large batch read
    batch_ids = list(large_metadata.keys())
    batch_result = temp_sqlite_store.get_batch(batch_ids)
    
    assert len(batch_result) == len(large_metadata)
    for record_id in batch_ids:
        assert batch_result[record_id] == large_metadata[record_id]
