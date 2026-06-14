"""
Test suite for token optimization changes.
Ensures backward compatibility and validates token reduction.
"""

import json
from unittest.mock import Mock

import pytest

from src.server.services.projects import ProjectService
from src.server.services.projects.document_service import DocumentService
from src.server.services.projects.task_service import TaskService
from src.server.services.storage import QueryResult


class _FakeQuery:
    """Fluent query stub that ignores filters and returns canned rows on execute()."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, _name):
        # Every builder method (select/insert/eq/order/include_unarchived/...) chains.
        def chain(*_args, **_kwargs):
            return self

        return chain

    async def execute(self):
        return QueryResult(data=self._data)


class FakeBackend:
    """Database backend stub: table() returns rows regardless of the query built."""

    def __init__(self, data):
        self._data = data
        self.table_calls = 0

    def table(self, _name):
        self.table_calls += 1
        return _FakeQuery(self._data)


class TestProjectServiceOptimization:
    """Test ProjectService with include_content parameter."""
    
    async def test_list_projects_with_full_content(self):
        """Test backward compatibility - default returns full content."""
        rows = [{
            "id": "test-id",
            "title": "Test Project",
            "description": "Test Description",
            "github_repo": "https://github.com/test/repo",
            "docs": [{"id": "doc1", "content": {"large": "content" * 100}}],
            "features": [{"feature1": "data"}],
            "data": [{"key": "value"}],
            "pinned": False,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01"
        }]

        backend = FakeBackend(rows)
        service = ProjectService(backend)
        success, result = await service.list_projects()  # Default include_content=True

        assert success
        assert len(result["projects"]) == 1
        assert "docs" in result["projects"][0]
        assert "features" in result["projects"][0]
        assert "data" in result["projects"][0]

        # Verify full content is returned
        assert len(result["projects"][0]["docs"]) == 1
        assert result["projects"][0]["docs"][0]["content"]["large"] is not None

    async def test_list_projects_lightweight(self):
        """Test lightweight response excludes large fields."""
        rows = [{
            "id": "test-id",
            "title": "Test Project",
            "description": "Test Description",
            "github_repo": "https://github.com/test/repo",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "pinned": False,
            "docs": [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}],  # 3 docs
            "features": [{"feature1": "data"}, {"feature2": "data"}],  # 2 features
            "data": [{"key": "value"}]  # Has data
        }]

        backend = FakeBackend(rows)
        service = ProjectService(backend)
        success, result = await service.list_projects(include_content=False)

        assert success
        assert len(result["projects"]) == 1
        project = result["projects"][0]

        # Verify no large fields
        assert "docs" not in project
        assert "features" not in project
        assert "data" not in project

        # Verify stats are present
        assert "stats" in project
        assert project["stats"]["docs_count"] == 3
        assert project["stats"]["features_count"] == 2
        assert project["stats"]["has_data"] is True

        # After the N+1 fix everything comes from a single query
        assert backend.table_calls == 1
    
    def test_token_reduction(self):
        """Verify token count reduction."""
        # Simulate full content response
        full_content = {
            "projects": [{
                "id": "test",
                "title": "Test",
                "description": "Test Description",
                "docs": [{"content": {"large": "x" * 10000}} for _ in range(5)],
                "features": [{"data": "y" * 5000} for _ in range(3)],
                "data": [{"values": "z" * 8000}]
            }]
        }
        
        # Simulate lightweight response
        lightweight = {
            "projects": [{
                "id": "test",
                "title": "Test",
                "description": "Test Description",
                "stats": {
                    "docs_count": 5,
                    "features_count": 3,
                    "has_data": True
                }
            }]
        }
        
        # Calculate approximate token counts (rough estimate: 1 token ≈ 4 chars)
        full_tokens = len(json.dumps(full_content)) / 4
        light_tokens = len(json.dumps(lightweight)) / 4
        
        reduction_percentage = (1 - light_tokens / full_tokens) * 100
        
        # Assert 95% reduction (allowing some margin)
        assert reduction_percentage > 95, f"Token reduction is only {reduction_percentage:.1f}%"


class TestTaskServiceOptimization:
    """Test TaskService with exclude_large_fields parameter."""
    
    async def test_list_tasks_with_large_fields(self):
        """Test backward compatibility - default includes large fields."""
        rows = [{
            "id": "task-1",
            "project_id": "proj-1",
            "title": "Test Task",
            "description": "Test Description",
            "sources": [{"url": "http://example.com", "content": "large"}],
            "code_examples": [{"code": "function() { /* large */ }"}],
            "status": "todo",
            "assignee": "User",
            "task_order": 0,
            "feature": None,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01"
        }]

        service = TaskService(FakeBackend(rows))
        success, result = await service.list_tasks()

        assert success
        assert "sources" in result["tasks"][0]
        assert "code_examples" in result["tasks"][0]

    async def test_list_tasks_exclude_large_fields(self):
        """Test excluding large fields returns counts instead."""
        rows = [{
            "id": "task-1",
            "project_id": "proj-1",
            "title": "Test Task",
            "description": "Test Description",
            "status": "todo",
            "assignee": "User",
            "task_order": 0,
            "feature": None,
            "sources": [1, 2, 3],  # Will be counted
            "code_examples": [1, 2],  # Will be counted
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01"
        }]

        service = TaskService(FakeBackend(rows))
        success, result = await service.list_tasks(exclude_large_fields=True)

        assert success
        task = result["tasks"][0]
        assert "sources" not in task
        assert "code_examples" not in task
        assert "stats" in task
        assert task["stats"]["sources_count"] == 3
        assert task["stats"]["code_examples_count"] == 2


class TestDocumentServiceOptimization:
    """Test DocumentService with include_content parameter."""
    
    async def test_list_documents_metadata_only(self):
        """Test default returns metadata only."""
        rows = [{
            "docs": [{
                "id": "doc-1",
                "title": "Test Doc",
                "content": {"huge": "content" * 1000},
                "document_type": "spec",
                "status": "draft",
                "version": "1.0",
                "tags": ["test"],
                "author": "Test Author"
            }]
        }]

        service = DocumentService(FakeBackend(rows))
        success, result = await service.list_documents("project-1")  # Default include_content=False

        assert success
        doc = result["documents"][0]
        assert "content" not in doc
        assert "stats" in doc
        assert doc["stats"]["content_size"] > 0
        assert doc["title"] == "Test Doc"

    async def test_list_documents_with_content(self):
        """Test include_content=True returns full documents."""
        rows = [{
            "docs": [{
                "id": "doc-1",
                "title": "Test Doc",
                "content": {"huge": "content"},
                "document_type": "spec"
            }]
        }]

        service = DocumentService(FakeBackend(rows))
        success, result = await service.list_documents("project-1", include_content=True)

        assert success
        doc = result["documents"][0]
        assert "content" in doc
        assert doc["content"]["huge"] == "content"


class TestBackwardCompatibility:
    """Ensure all changes are backward compatible."""
    
    def test_api_defaults_preserve_behavior(self):
        """Test that API defaults maintain current behavior."""
        # ProjectService default should include content
        service = ProjectService(Mock())
        # Check default parameter value
        import inspect
        sig = inspect.signature(service.list_projects)
        assert sig.parameters['include_content'].default is True
        
        # DocumentService default should NOT include content
        doc_service = DocumentService(Mock())
        sig = inspect.signature(doc_service.list_documents)
        assert sig.parameters['include_content'].default is False
        
        # TaskService default should NOT exclude fields
        task_service = TaskService(Mock())
        sig = inspect.signature(task_service.list_tasks)
        assert sig.parameters['exclude_large_fields'].default is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])