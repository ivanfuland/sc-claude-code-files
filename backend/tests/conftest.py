"""
Test configuration and fixtures for RAG Chatbot test suite.

This module provides shared fixtures and test data for all tests.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Import the modules we're testing
import sys
sys.path.append('..')  # Add parent directory to path

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


@pytest.fixture
def mock_config():
    """Provide a test configuration with safe test values"""
    config = Config()
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key-12345"
    config.ANTHROPIC_MODEL = "claude-test-model"
    config.EMBEDDING_MODEL = "test-embedding-model"
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_lesson_1():
    """Create a sample lesson for testing"""
    return Lesson(
        lesson_number=1,
        title="Introduction to Python Basics",
        lesson_link="https://example.com/lesson1"
    )


@pytest.fixture
def sample_lesson_2():
    """Create another sample lesson for testing"""
    return Lesson(
        lesson_number=2,
        title="Variables and Data Types",
        lesson_link="https://example.com/lesson2"
    )


@pytest.fixture
def sample_course(sample_lesson_1, sample_lesson_2):
    """Create a sample course with lessons for testing"""
    return Course(
        title="Python Programming Fundamentals",
        course_link="https://example.com/course",
        instructor="John Doe",
        lessons=[sample_lesson_1, sample_lesson_2]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language that is easy to learn.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python can store different types of data like strings, numbers, and booleans.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Data types in Python include int, float, str, bool, and complex.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_search_results():
    """Create mock search results for testing"""
    return SearchResults(
        documents=["Python is a programming language", "Variables store data"],
        metadata=[
            {"course_title": "Python Programming Fundamentals", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Python Programming Fundamentals", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create search results with error for testing"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Create a mock VectorStore for testing"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search.return_value = mock_search_results
    mock_store.get_existing_course_titles.return_value = ["Python Programming Fundamentals"]
    mock_store.get_course_count.return_value = 1
    mock_store.get_all_courses_metadata.return_value = [{
        "title": "Python Programming Fundamentals",
        "instructor": "John Doe",
        "course_link": "https://example.com/course",
        "lesson_count": 2
    }]
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock response for direct text response
    mock_text_response = Mock()
    mock_text_response.stop_reason = "end_turn"
    mock_text_response.content = [Mock(text="This is a test response from Claude.")]
    
    # Mock response for tool use
    mock_tool_response = Mock()
    mock_tool_response.stop_reason = "tool_use"
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_call_123"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_response.content = [mock_tool_content]
    
    # Configure the mock to return different responses
    mock_client.messages.create.return_value = mock_text_response
    
    return mock_client


@pytest.fixture
def mock_course_search_tool(mock_vector_store):
    """Create a mock CourseSearchTool for testing"""
    tool = CourseSearchTool(mock_vector_store)
    return tool


@pytest.fixture
def mock_tool_manager(mock_course_search_tool):
    """Create a mock ToolManager with registered CourseSearchTool"""
    manager = ToolManager()
    manager.register_tool(mock_course_search_tool)
    return manager


@pytest.fixture
def mock_ai_generator(mock_anthropic_client, mock_config):
    """Create a mock AIGenerator for testing"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        return generator


@pytest.fixture
def sample_query_responses():
    """Sample queries and expected responses for testing"""
    return {
        "basic_query": {
            "query": "What is Python?",
            "expected_type": "general_knowledge"
        },
        "course_specific_query": {
            "query": "What are the data types in Python Programming Fundamentals?",
            "expected_type": "course_specific",
            "expected_course": "Python Programming Fundamentals"
        },
        "lesson_specific_query": {
            "query": "What is covered in lesson 2 of Python course?",
            "expected_type": "lesson_specific",
            "expected_course": "Python Programming Fundamentals",
            "expected_lesson": 2
        }
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Auto-cleanup fixture to remove test files after each test"""
    yield
    # Clean up any test files
    test_files = ["test_chroma_db", "test_data.json", "test_logs.txt"]
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
            else:
                os.remove(file_path)


# Mock data for edge cases and error testing
@pytest.fixture
def malformed_search_results():
    """Create malformed search results to test error handling"""
    return {
        'documents': [[]],  # Wrong structure
        'metadatas': None,  # Missing metadata
        'distances': [[]]   # Wrong structure
    }


@pytest.fixture
def large_search_results():
    """Create large search results to test pagination/limits"""
    documents = [f"Document {i} content" for i in range(100)]
    metadata = [{"course_title": "Large Course", "lesson_number": i % 10, "chunk_index": i} for i in range(100)]
    distances = [0.1 * i for i in range(100)]
    
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=distances
    )


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Create data for performance testing"""
    courses = []
    chunks = []
    
    for i in range(10):  # 10 courses
        course = Course(
            title=f"Course {i}",
            instructor=f"Instructor {i}",
            course_link=f"https://example.com/course{i}",
            lessons=[
                Lesson(lesson_number=j, title=f"Lesson {j}", lesson_link=f"https://example.com/lesson{i}_{j}")
                for j in range(1, 6)  # 5 lessons per course
            ]
        )
        courses.append(course)
        
        # Create chunks for each course
        for lesson_num in range(1, 6):
            for chunk_idx in range(5):  # 5 chunks per lesson
                chunk = CourseChunk(
                    content=f"Content for course {i}, lesson {lesson_num}, chunk {chunk_idx}",
                    course_title=course.title,
                    lesson_number=lesson_num,
                    chunk_index=chunk_idx + (lesson_num - 1) * 5
                )
                chunks.append(chunk)
    
    return {"courses": courses, "chunks": chunks}