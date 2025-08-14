"""
Unit tests for CourseSearchTool and ToolManager classes.

Tests the execute method of CourseSearchTool with various scenarios
including successful searches, empty results, errors, and edge cases.
"""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from search_tools import CourseSearchTool, Tool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool class"""

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_successful_basic_search(self, mock_search_results):
        """Test successful search with only query parameter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is Python?")

        mock_vector_store.search.assert_called_once_with(
            query="What is Python?", course_name=None, lesson_number=None
        )

        # Check that result contains formatted content
        assert "[Python Programming Fundamentals - Lesson 1]" in result
        assert "Python is a programming language" in result
        assert "[Python Programming Fundamentals - Lesson 2]" in result
        assert "Variables store data" in result

        # Check that sources are tracked
        assert len(tool.last_sources) == 2
        assert "Python Programming Fundamentals - Lesson 1" in tool.last_sources
        assert "Python Programming Fundamentals - Lesson 2" in tool.last_sources

    def test_execute_successful_course_filtered_search(self, mock_search_results):
        """Test successful search with course name filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="variables", course_name="Python Programming")

        mock_vector_store.search.assert_called_once_with(
            query="variables", course_name="Python Programming", lesson_number=None
        )

        assert "Python Programming Fundamentals" in result
        assert len(tool.last_sources) > 0

    def test_execute_successful_lesson_filtered_search(self, mock_search_results):
        """Test successful search with lesson number filter"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="data types", course_name="Python Programming", lesson_number=2
        )

        mock_vector_store.search.assert_called_once_with(
            query="data types", course_name="Python Programming", lesson_number=2
        )

        assert "Python Programming Fundamentals" in result

    def test_execute_empty_results(self, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent content")

        assert result == "No relevant content found."
        assert tool.last_sources == []

    def test_execute_empty_results_with_course_filter(self, empty_search_results):
        """Test empty results with course filter shows filter info"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent", course_name="Unknown Course")

        assert result == "No relevant content found in course 'Unknown Course'."

    def test_execute_empty_results_with_lesson_filter(self, empty_search_results):
        """Test empty results with lesson filter shows filter info"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent", lesson_number=5)

        assert result == "No relevant content found in lesson 5."

    def test_execute_empty_results_with_both_filters(self, empty_search_results):
        """Test empty results with both filters shows both in message"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="nonexistent", course_name="Test Course", lesson_number=3
        )

        assert (
            result == "No relevant content found in course 'Test Course' in lesson 3."
        )

    def test_execute_error_handling(self, error_search_results):
        """Test handling of search errors"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        assert result == "Database connection failed"
        assert tool.last_sources == []

    def test_execute_vector_store_exception(self):
        """Test handling when vector store raises exception"""
        mock_vector_store = Mock()
        mock_vector_store.search.side_effect = Exception("Connection timeout")

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        # Should handle gracefully and return error message
        assert "error" in result.lower() or "failed" in result.lower()

    def test_format_results_no_lesson_number(self):
        """Test result formatting when lesson_number is None"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        search_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course", "chunk_index": 0}],
            distances=[0.1],
        )

        result = tool._format_results(search_results)

        assert "[Test Course]" in result
        assert "Content without lesson number" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0] == "Test Course"

    def test_format_results_with_lesson_number(self):
        """Test result formatting when lesson_number is provided"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        search_results = SearchResults(
            documents=["Content with lesson number"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 3, "chunk_index": 0}
            ],
            distances=[0.1],
        )

        result = tool._format_results(search_results)

        assert "[Test Course - Lesson 3]" in result
        assert "Content with lesson number" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0] == "Test Course - Lesson 3"

    def test_format_results_multiple_documents(self):
        """Test formatting multiple documents"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        search_results = SearchResults(
            documents=["First document", "Second document"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Course B", "lesson_number": 2, "chunk_index": 1},
            ],
            distances=[0.1, 0.2],
        )

        result = tool._format_results(search_results)

        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First document" in result
        assert "Second document" in result
        assert "\n\n" in result  # Documents should be separated by double newline
        assert len(tool.last_sources) == 2

    def test_format_results_missing_metadata(self):
        """Test handling of missing metadata fields"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        search_results = SearchResults(
            documents=["Document with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
        )

        result = tool._format_results(search_results)

        assert "[unknown]" in result
        assert "Document with missing metadata" in result

    def test_last_sources_reset_between_searches(
        self, mock_search_results, empty_search_results
    ):
        """Test that last_sources is properly updated between searches"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        # First search with results
        mock_vector_store.search.return_value = mock_search_results
        tool.execute(query="first query")
        assert len(tool.last_sources) > 0

        # Second search with no results
        mock_vector_store.search.return_value = empty_search_results
        tool.execute(query="second query")
        assert len(tool.last_sources) == 0


class TestToolManager:
    """Test cases for ToolManager class"""

    def test_register_tool_success(self):
        """Test successful tool registration"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test",
        }

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_register_tool_without_name(self):
        """Test error when registering tool without name"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "description": "Test"
        }  # Missing name

        with pytest.raises(
            ValueError, match="Tool must have a 'name' in its definition"
        ):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        mock_tool1 = Mock(spec=Tool)
        mock_tool2 = Mock(spec=Tool)

        mock_tool1.get_tool_definition.return_value = {
            "name": "tool1",
            "description": "First tool",
        }
        mock_tool2.get_tool_definition.return_value = {
            "name": "tool2",
            "description": "Second tool",
        }

        manager.register_tool(mock_tool1)
        manager.register_tool(mock_tool2)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert {"name": "tool1", "description": "First tool"} in definitions
        assert {"name": "tool2", "description": "Second tool"} in definitions

    def test_execute_tool_success(self):
        """Test successful tool execution"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test",
        }
        mock_tool.execute.return_value = "Tool executed successfully"

        manager.register_tool(mock_tool)
        result = manager.execute_tool("test_tool", param1="value1", param2="value2")

        assert result == "Tool executed successfully"
        mock_tool.execute.assert_called_once_with(param1="value1", param2="value2")

    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", param="value")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources_from_search_tool(self, mock_search_results):
        """Test getting sources from registered CourseSearchTool"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert "Python Programming Fundamentals - Lesson 1" in sources

    def test_get_last_sources_no_search_tools(self):
        """Test getting sources when no search tools are registered"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "non_search_tool",
            "description": "Test",
        }

        manager.register_tool(mock_tool)

        sources = manager.get_last_sources()
        assert sources == []

    def test_reset_sources(self, mock_search_results):
        """Test resetting sources from all registered tools"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()
        assert manager.get_last_sources() == []
        assert search_tool.last_sources == []

    def test_reset_sources_multiple_tools(self, mock_search_results):
        """Test resetting sources from multiple tools"""
        manager = ToolManager()
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_search_results

        # Register multiple search tools
        search_tool1 = CourseSearchTool(mock_vector_store)
        search_tool2 = CourseSearchTool(mock_vector_store)

        # Create a custom tool definition for the second tool
        original_def = {
            "name": "search_course_content_2",
            "description": "Second search tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        }
        search_tool2.get_tool_definition = Mock(return_value=original_def)

        manager.register_tool(search_tool1)
        manager.register_tool(search_tool2)

        # Execute searches to populate sources
        manager.execute_tool("search_course_content", query="test1")
        manager.execute_tool("search_course_content_2", query="test2")

        # Both tools should have sources
        assert len(search_tool1.last_sources) > 0
        assert len(search_tool2.last_sources) > 0

        # Reset should clear both
        manager.reset_sources()
        assert search_tool1.last_sources == []
        assert search_tool2.last_sources == []


class TestToolEdgeCases:
    """Test edge cases and error conditions"""

    def test_course_search_tool_with_none_vector_store(self):
        """Test CourseSearchTool behavior with None vector store"""
        # Tool creation should work, but execution should fail
        tool = CourseSearchTool(None)

        # Execution should return an error message due to try/catch in execute method
        result = tool.execute(query="test")
        assert "Search error:" in result

    def test_execute_with_empty_query(self):
        """Test execution with empty query"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults.empty("Empty query")

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="")

        # Should still call search but may return empty results
        mock_vector_store.search.assert_called_once()

    def test_execute_with_invalid_lesson_number(self):
        """Test execution with invalid lesson number"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults.empty("Invalid lesson")

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", lesson_number=-1)

        # Should pass through to vector store, let it handle validation
        mock_vector_store.search.assert_called_once_with(
            query="test", course_name=None, lesson_number=-1
        )

    def test_format_results_mismatched_lengths(self):
        """Test format results with mismatched document/metadata lengths"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        # Mismatched lengths - more documents than metadata
        search_results = SearchResults(
            documents=["Doc1", "Doc2", "Doc3"],
            metadata=[{"course_title": "Course1"}],  # Only one metadata
            distances=[0.1, 0.2, 0.3],
        )

        # Should handle gracefully without crashing
        result = tool._format_results(search_results)
        assert "Doc1" in result
        # Should not crash even with mismatched data
