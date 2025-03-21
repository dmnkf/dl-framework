import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import sqlite3

import torch
from numpy import ndarray

logger = logging.getLogger(__name__)


class MetadataStoreBackendBase(ABC):
    """Abstract base class for handling metadata storage operations"""

    @abstractmethod
    def get(self, record_id: str) -> dict:
        pass

    @abstractmethod
    def set(self, metadata_dict: Dict[str, dict], overwrite: bool = False):
        pass

    def store_single(self, record_id: str, metadata: dict, overwrite: bool = False):
        self.set({record_id: metadata}, overwrite)

    @abstractmethod
    def get_batch(self, record_ids: list) -> dict:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_all_metadata(self) -> dict:
        pass


class JSONMetadataStoreBackend(MetadataStoreBackendBase):
    """Class responsible for handling the JSON file operations for metadata storage"""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._lock = threading.Lock()

    def _get_metadata_file_path(self) -> Path:
        return self.data_root / "metadata.json"

    def get(self, record_id: str) -> dict:
        """Get metadata for a record from stored preprocessed data"""
        metadata = self.get_all_metadata()
        return metadata.get(record_id, {})

    def get_all_metadata(self) -> dict:
        """Get all metadata for a dataset"""
        with self._lock:
            metadata_file = self._get_metadata_file_path()
            if not metadata_file.exists():
                return {}
            with open(metadata_file, "r") as f:
                return json.load(f)

    def set(self, metadata_dict: Dict[str, Any], overwrite: bool = False):
        """Store metadata for an entire dataset with thread-safe file access"""
        existing_metadata = self.get_all_metadata()

        for record_id, record_metadata in metadata_dict.items():
            if record_id in existing_metadata and not overwrite:
                existing_metadata[record_id].update(record_metadata)
            else:
                existing_metadata[record_id] = record_metadata

        with self._lock:
            with open(self._get_metadata_file_path(), "w") as f:
                json.dump(existing_metadata, f, indent=2)

    def get_batch(self, record_ids: list) -> dict:
        """Get metadata for multiple records efficiently"""
        metadata = self.get_all_metadata()
        return {record_id: metadata.get(record_id, {}) for record_id in record_ids}

    def clear(self):
        """Clear metadata for the dataset"""
        logger.info("Starting JSON metadata cleanup...")
        with self._lock:
            if self._get_metadata_file_path().exists():
                self._get_metadata_file_path().unlink()
            logger.info("JSON metadata file removed successfully")


class SQLiteMetadataStoreBackend(MetadataStoreBackendBase):
    """SQLite implementation with thread safety and batch operations"""
    
    def __init__(self, data_root: Path):
        self.db_path = data_root / "metadata.db"
        # Use RLock to allow nested lock acquisition (e.g. clear() calling _init_table())
        self._lock = threading.RLock()
        logger.info(f"Opening SQLite connection to {self.db_path}")
        try:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30
            )
            
            # Configure for maximum thread safety
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            logger.info("SQLite connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to establish SQLite connection: {e}")
            if hasattr(self, 'conn'):
                self.conn.close()
            raise
        self._init_table()  # This is safe now with RLock

    def _init_table(self):
        logger.info("Initializing SQLite table...")
        with self._lock:
            try:
                self.conn.execute('''CREATE TABLE IF NOT EXISTS metadata (
                               id TEXT PRIMARY KEY,
                               data TEXT)''')
                self.conn.commit()
                logger.info("Table initialization completed")
            except Exception as e:
                logger.error(f"Error during table initialization: {e}")
                raise

    def get(self, record_id: str, default=None) -> dict:
        with self._lock:
            cursor = self.conn.execute(
                "SELECT data FROM metadata WHERE id = ?",
                (record_id,)
            )
            result = cursor.fetchone()
            # Return the data column (which contains the JSON metadata)
            return json.loads(result[0]) if result else {}

    def set(self, metadata_dict: Dict[str, Any], overwrite: bool = False):
        with self._lock:  # Single lock for the entire operation
            try:
                self.conn.execute("BEGIN TRANSACTION")
                
                for record_id, metadata in metadata_dict.items():
                    if not overwrite:
                        # Get existing within the same transaction
                        cursor = self.conn.execute(
                            "SELECT data FROM metadata WHERE id = ?",
                            (record_id,)
                        )
                        result = cursor.fetchone()
                        if result:
                            existing = json.loads(result[0])
                            existing.update(metadata)
                            metadata = existing
                    
                    self.conn.execute(
                        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                        (record_id, json.dumps(metadata))
                    )
                
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise

    def get_batch(self, record_ids: list) -> dict:
        with self._lock:
            placeholders = ','.join(['?']*len(record_ids))
            cursor = self.conn.execute(
                f"SELECT id, data FROM metadata WHERE id IN ({placeholders})",
                record_ids
            )
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def clear(self):
        """Clear metadata for the dataset"""
        logger.info("Starting SQLite database reset...")
        with self._lock:
            try:
                # Close the current connection
                self.conn.close()
                logger.info("Closed existing connection")
                
                # Remove the database file
                if self.db_path.exists():
                    self.db_path.unlink()
                logger.info("Removed database file")
                
                # Create new connection and table
                self.conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30
                )
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self.conn.execute("PRAGMA busy_timeout=30000")
                logger.info("Created new connection")
                
                self._init_table()
            except Exception as e:
                logger.error(f"Error during SQLite clear operation: {e}")
                raise

    def get_all_metadata(self) -> dict:
        with self._lock:
            cursor = self.conn.execute("SELECT id, data FROM metadata")
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def export_to_json(self, output_path: Path):
        """Export all metadata to a JSON file"""
        logger.info(f"Exporting metadata to JSON file: {output_path}")
        with self._lock:
            try:
                metadata = self.get_all_metadata()
                with open(output_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Metadata export completed successfully")
            except Exception as e:
                logger.error(f"Error during metadata export: {e}")
                raise

    def __del__(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
                logger.info("SQLite connection closed in cleanup")
        except:
            pass


class MetadataStore:
    """Registry for accessing metadata from preprocessed datasets using a backend store"""

    def __init__(
        self,
        data_root: Path,
        store: Optional[MetadataStoreBackendBase] = None,
    ):
        self.data_root = data_root

        if store is None:
            logger.info(f"Initializing SQLite metadata store at {self.data_root}")
            self.backend = SQLiteMetadataStoreBackend(self.data_root)
        else:
            self.backend = store

    def get(self, record_id: str, default=None) -> dict:
        """Get metadata for a record from stored preprocessed data"""
        return self.backend.get(record_id)

    def get_batch(self, record_ids: list) -> dict:
        """Get metadata for multiple records efficiently"""
        return self.backend.get_batch(record_ids)

    def set(self, metadata_dict: Dict[str, dict], overwrite: bool = False):
        """Store metadata for an entire dataset."""
        for key, value in metadata_dict.items():
            for k, v in value.items():
                if isinstance(v, ndarray) or isinstance(v, torch.Tensor):
                    value[k] = v.tolist()
        self.backend.set(metadata_dict, overwrite)

    def add(self, record_id: str, metadata: dict, overwrite: bool = False):
        """Store metadata for a single record"""
        self.backend.store_single(record_id, metadata, overwrite)

    def reset(self):
        """Reset metadata for the dataset"""
        logger.info("Starting metadata store reset...")
        try:
            self.backend.clear()
            logger.info("Metadata store reset completed successfully")
        except Exception as e:
            logger.error(f"Error during metadata store reset: {e}")
            raise

    def available_fields(self) -> set:
        """Get all available metadata fields"""
        return set(
            field
            for record in self.backend.get_all_metadata().values()
            for field in record.keys()
        )
