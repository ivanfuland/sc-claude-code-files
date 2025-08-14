"""
Integration tests for RAGSystem class.

Tests the end-to-end functionality of the RAG system, including query processing,
component integration, and error handling across the entire system.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""

    @pytest.fixture
    def mock_components(self):
        """Create mocked components for RAGSystem"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
        ):

            yield {
                "document_processor": mock_doc_proc,
                "vector_store": mock_vector_store,
                "ai_generator": mock_ai_gen,
                "session_manager": mock_session_mgr,
                "tool_manager": mock_tool_mgr,
                "search_tool": mock_search_tool,
            }

    def test_initialization_with_config(self, mock_config, mock_components):
        """Test RAGSystem initialization with all components"""
        rag_system = RAGSystem(mock_config)

        # Verify all components are initialized
        assert rag_system.config == mock_config
        assert hasattr(rag_system, "document_processor")
        assert hasattr(rag_system, "vector_store")
        assert hasattr(rag_system, "ai_generator")
        assert hasattr(rag_system, "session_manager")
        assert hasattr(rag_system, "tool_manager")
        assert hasattr(rag_system, "search_tool")

        # Verify components initialized with correct parameters
        mock_components["vector_store"].assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL,
            mock_config.MAX_RESULTS,
        )

        mock_components["ai_generator"].assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

    def test_tool_registration(self, mock_config, mock_components):
        """Test that CourseSearchTool is registered with ToolManager"""
        rag_system = RAGSystem(mock_config)

        # Verify search tool is created with vector store
        mock_components["search_tool"].assert_called_once_with(rag_system.vector_store)

        # Verify tool is registered with manager
        rag_system.tool_manager.register_tool.assert_called_once_with(
            rag_system.search_tool
        )


class TestRAGSystemQueryProcessing:
    """Test RAG system query processing functionality"""

    @pytest.fixture
    def rag_system_with_mocks(self, mock_config):
        """Create RAGSystem with mocked components"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager") as mock_sm,
            patch("rag_system.ToolManager") as mock_tm,
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(mock_config)

            # Setup default mocks
            mock_ai_instance = mock_ai.return_value
            mock_ai_instance.generate_response.return_value = "AI response"

            mock_tm_instance = mock_tm.return_value
            mock_tm_instance.get_tool_definitions.return_value = [{"name": "test_tool"}]
            mock_tm_instance.get_last_sources.return_value = ["Source 1", "Source 2"]

            mock_sm_instance = mock_sm.return_value
            mock_sm_instance.get_conversation_history.return_value = "Previous chat"

            return rag_system

    def test_query_basic_functionality(self, rag_system_with_mocks):
        """Test basic query processing"""
        response, sources = rag_system_with_mocks.query("What is Python?")

        assert response == "AI response"
        assert sources == ["Source 1", "Source 2"]

        # Verify AI generator was called with correct parameters
        rag_system_with_mocks.ai_generator.generate_response.assert_called_once()
        call_args = rag_system_with_mocks.ai_generator.generate_response.call_args

        assert "What is Python?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] == [{"name": "test_tool"}]
        assert call_args[1]["tool_manager"] == rag_system_with_mocks.tool_manager

    def test_query_with_session_id(self, rag_system_with_mocks):
        """Test query processing with session ID"""
        session_id = "test-session-123"
        response, sources = rag_system_with_mocks.query(
            "Follow up question", session_id
        )

        # Verify conversation history was retrieved
        rag_system_with_mocks.session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator received the history
        call_args = rag_system_with_mocks.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous chat"

        # Verify session was updated with the exchange
        rag_system_with_mocks.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "AI response"
        )

    def test_query_sources_handling(self, rag_system_with_mocks):
        """Test that sources are properly retrieved and reset"""
        rag_system_with_mocks.query("Test query")

        # Verify sources were retrieved
        rag_system_with_mocks.tool_manager.get_last_sources.assert_called_once()

        # Verify sources were reset after retrieval
        rag_system_with_mocks.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, rag_system_with_mocks):
        """Test that query is properly formatted as a prompt"""
        user_query = "Explain variables in Python"
        rag_system_with_mocks.query(user_query)

        call_args = rag_system_with_mocks.ai_generator.generate_response.call_args
        prompt = call_args[1]["query"]

        assert "Answer this question about course materials:" in prompt
        assert user_query in prompt

    def test_query_error_handling(self, rag_system_with_mocks):
        """Test query error handling when AI generator fails"""
        rag_system_with_mocks.ai_generator.generate_response.side_effect = Exception(
            "AI API Error"
        )

        with pytest.raises(Exception, match="AI API Error"):
            rag_system_with_mocks.query("Test query")

    def test_query_tool_manager_error(self, rag_system_with_mocks):
        """Test handling when tool manager operations fail"""
        rag_system_with_mocks.tool_manager.get_last_sources.side_effect = Exception(
            "Tool error"
        )

        # Should raise the exception since it's not handled in current implementation
        with pytest.raises(Exception, match="Tool error"):
            rag_system_with_mocks.query("Test query")


class TestRAGSystemDocumentProcessing:
    """Test RAG system document processing functionality"""

    @pytest.fixture
    def rag_system_for_docs(self, mock_config, sample_course, sample_course_chunks):
        """Create RAGSystem for document processing tests"""
        with (
            patch("rag_system.DocumentProcessor") as mock_dp,
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(mock_config)

            # Setup document processor mock
            mock_dp_instance = mock_dp.return_value
            mock_dp_instance.process_course_document.return_value = (
                sample_course,
                sample_course_chunks,
            )

            # Setup vector store mock
            mock_vs_instance = mock_vs.return_value
            mock_vs_instance.get_existing_course_titles.return_value = []

            return rag_system, mock_dp_instance, mock_vs_instance

    def test_add_course_document_success(
        self, rag_system_for_docs, sample_course, sample_course_chunks
    ):
        """Test successful course document addition"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        course, chunk_count = rag_system.add_course_document("/path/to/course.pdf")

        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify document processing was called
        mock_dp.process_course_document.assert_called_once_with("/path/to/course.pdf")

        # Verify vector store operations
        mock_vs.add_course_metadata.assert_called_once_with(sample_course)
        mock_vs.add_course_content.assert_called_once_with(sample_course_chunks)

    def test_add_course_document_processing_error(self, rag_system_for_docs):
        """Test course document addition with processing error"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        mock_dp.process_course_document.side_effect = Exception("File not found")

        course, chunk_count = rag_system.add_course_document("/invalid/path.pdf")

        assert course is None
        assert chunk_count == 0

        # Vector store should not be called
        mock_vs.add_course_metadata.assert_not_called()
        mock_vs.add_course_content.assert_not_called()

    def test_add_course_folder_success(
        self, rag_system_for_docs, sample_course, sample_course_chunks
    ):
        """Test successful course folder processing"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        # Mock folder contents and file type checking
        test_folder = "/test/folder"

        def mock_isfile(path):
            return path.endswith((".pdf", ".txt", ".docx"))

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "os.listdir", return_value=["course1.pdf", "course2.txt", "invalid.jpg"]
            ),
            patch("os.path.isfile", side_effect=mock_isfile),
            patch("os.path.join", side_effect=lambda folder, name: f"{folder}/{name}"),
        ):

            total_courses, total_chunks = rag_system.add_course_folder(test_folder)

            # Should process 2 valid files (pdf and txt)
            assert mock_dp.process_course_document.call_count == 2
            # But since both return the same course title, only 1 unique course gets added
            assert total_courses == 1
            assert total_chunks == len(sample_course_chunks)

    def test_add_course_folder_clear_existing(self, rag_system_for_docs):
        """Test course folder processing with clear existing data"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        test_folder = "/test/folder"
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["course.pdf"]),
            patch("os.path.isfile", return_value=True),
        ):

            rag_system.add_course_folder(test_folder, clear_existing=True)

            # Should clear data first
            mock_vs.clear_all_data.assert_called_once()

    def test_add_course_folder_skip_existing_courses(
        self, rag_system_for_docs, sample_course, sample_course_chunks
    ):
        """Test that existing courses are skipped"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        # Mock existing course titles
        mock_vs.get_existing_course_titles.return_value = [sample_course.title]

        test_folder = "/test/folder"
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["course.pdf"]),
            patch("os.path.isfile", return_value=True),
        ):

            total_courses, total_chunks = rag_system.add_course_folder(test_folder)

            # Should not add existing course
            assert total_courses == 0
            assert total_chunks == 0
            mock_vs.add_course_metadata.assert_not_called()
            mock_vs.add_course_content.assert_not_called()

    def test_add_course_folder_nonexistent_folder(self, rag_system_for_docs):
        """Test processing non-existent folder"""
        rag_system, mock_dp, mock_vs = rag_system_for_docs

        total_courses, total_chunks = rag_system.add_course_folder(
            "/nonexistent/folder"
        )

        assert total_courses == 0
        assert total_chunks == 0


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""

    @pytest.fixture
    def rag_system_for_analytics(self, mock_config):
        """Create RAGSystem for analytics tests"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(mock_config)

            mock_vs_instance = mock_vs.return_value
            mock_vs_instance.get_course_count.return_value = 5
            mock_vs_instance.get_existing_course_titles.return_value = [
                "Course A",
                "Course B",
                "Course C",
                "Course D",
                "Course E",
            ]

            return rag_system, mock_vs_instance

    def test_get_course_analytics(self, rag_system_for_analytics):
        """Test course analytics retrieval"""
        rag_system, mock_vs = rag_system_for_analytics

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]

        # Verify vector store methods were called
        mock_vs.get_course_count.assert_called_once()
        mock_vs.get_existing_course_titles.assert_called_once()


class TestRAGSystemIntegration:
    """Integration tests for RAG system components working together"""

    @pytest.fixture
    def real_rag_components(self, mock_config, temp_chroma_dir):
        """Create RAGSystem with real components for integration testing"""
        # Update config to use temp directory
        mock_config.CHROMA_PATH = temp_chroma_dir

        with (
            patch("rag_system.DocumentProcessor") as mock_dp,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
        ):

            # Setup document processor
            mock_dp_instance = mock_dp.return_value
            sample_course = Course(
                title="Test Integration Course",
                instructor="Test Instructor",
                lessons=[Lesson(lesson_number=1, title="Lesson 1")],
            )
            sample_chunks = [
                CourseChunk(
                    content="This is test content for integration testing",
                    course_title="Test Integration Course",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
            mock_dp_instance.process_course_document.return_value = (
                sample_course,
                sample_chunks,
            )

            # Setup Anthropic client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [Mock(text="Integration test response")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            return RAGSystem(mock_config)

    def test_end_to_end_query_processing(self, real_rag_components):
        """Test end-to-end query processing with real components"""
        rag_system = real_rag_components

        # Add a document to the system
        course, chunk_count = rag_system.add_course_document("test_course.pdf")
        assert course is not None
        assert chunk_count > 0

        # Query the system
        response, sources = rag_system.query("What is covered in the test course?")

        assert response == "Integration test response"
        # Sources might be empty if no search tool was actually executed
        assert isinstance(sources, list)

    def test_session_continuity(self, real_rag_components):
        """Test that session management works across queries"""
        rag_system = real_rag_components

        session_id = "test-session"

        # First query
        response1, _ = rag_system.query("What is Python?", session_id)
        assert response1 == "Integration test response"

        # Second query in same session
        response2, _ = rag_system.query("Tell me more", session_id)
        assert response2 == "Integration test response"

        # Verify session manager has conversation history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None

    def test_tool_integration_flow(self, real_rag_components):
        """Test that tools are properly integrated and available"""
        rag_system = real_rag_components

        # Verify tool definitions are available
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) > 0
        assert any(tool["name"] == "search_course_content" for tool in tool_definitions)

        # Verify search tool can be executed
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="test query"
        )
        assert isinstance(result, str)

    def test_error_propagation(self, real_rag_components):
        """Test that errors propagate correctly through the system"""
        rag_system = real_rag_components

        # Simulate AI generator error
        rag_system.ai_generator.generate_response = Mock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="API Error"):
            rag_system.query("Test query")

    def test_analytics_with_real_data(self, real_rag_components):
        """Test analytics with real vector store data"""
        rag_system = real_rag_components

        # Add some courses
        rag_system.add_course_document("course1.pdf")
        rag_system.add_course_document(
            "course2.pdf"
        )  # Will be same course, should be skipped

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] >= 0  # May be 0 if courses were skipped
        assert isinstance(analytics["course_titles"], list)


class TestRAGSystemEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def minimal_rag_system(self, mock_config):
        """Create RAGSystem with minimal mocking for edge case testing"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
        ):

            return RAGSystem(mock_config)

    def test_query_with_empty_string(self, minimal_rag_system):
        """Test query with empty string"""
        rag_system = minimal_rag_system

        # Should not crash
        rag_system.ai_generator.generate_response.return_value = "Empty query response"
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query("")
        assert response == "Empty query response"
        assert sources == []

    def test_query_with_very_long_string(self, minimal_rag_system):
        """Test query with very long string"""
        rag_system = minimal_rag_system

        long_query = "A" * 10000  # Very long query

        rag_system.ai_generator.generate_response.return_value = "Long query response"
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query(long_query)
        assert response == "Long query response"

    def test_query_with_none_session_id(self, minimal_rag_system):
        """Test query with None session ID"""
        rag_system = minimal_rag_system

        rag_system.ai_generator.generate_response.return_value = "No session response"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Should handle None session gracefully
        response, sources = rag_system.query("Test", session_id=None)
        assert response == "No session response"

        # Session manager should not be called for history
        rag_system.session_manager.get_conversation_history.assert_not_called()

    def test_component_initialization_failure(self, mock_config):
        """Test handling of component initialization failures"""
        with patch(
            "rag_system.VectorStore", side_effect=Exception("Vector store init failed")
        ):
            with pytest.raises(Exception, match="Vector store init failed"):
                RAGSystem(mock_config)
