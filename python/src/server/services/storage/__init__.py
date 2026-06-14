"""
Storage Services

Handles document and code storage operations.
"""

from .base_storage_service import BaseStorageService
from .code_storage_service import (
    add_code_examples_to_supabase,
    extract_code_blocks,
    generate_code_example_summary,
)
from .database_backend import DatabaseBackend
from .document_storage_service import add_documents_to_supabase
from .factory import (
    get_database_backend,
    reset_database_backend,
    set_database_backend,
)
from .postgres_backend import PostgresBackend
from .query_builder import PostgresQueryBuilder, QueryResult
from .storage_services import DocumentStorageService

__all__ = [
    # Base service
    "BaseStorageService",
    # Service classes
    "DocumentStorageService",
    # Document storage utilities
    "add_documents_to_supabase",
    # Code storage utilities
    "extract_code_blocks",
    "generate_code_example_summary",
    "add_code_examples_to_supabase",
    # Database backend abstraction
    "DatabaseBackend",
    "PostgresBackend",
    "PostgresQueryBuilder",
    "QueryResult",
    "get_database_backend",
    "set_database_backend",
    "reset_database_backend",
]
