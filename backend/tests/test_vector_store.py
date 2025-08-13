"""
Unit tests for VectorStore class.

Tests the search functionality, course management, and ChromaDB integration.
"""

import pytest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test cases for SearchResults dataclass"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results with data"""
        chroma_results = {
            'documents': [['Doc1', 'Doc2']],
            'metadatas': [[{'course': 'Course1'}, {'course': 'Course2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['Doc1', 'Doc2']
        assert results.metadata == [{'course': 'Course1'}, {'course': 'Course2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_with_empty_data(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_from_chroma_with_none_data(self):
        """Test creating SearchResults from ChromaDB results with None"""
        chroma_results = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=["Doc1"], 
            metadata=[{"course": "Course1"}], 
            distances=[0.1]
        )
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client, mock_collection
    
    @pytest.fixture
    def mock_vector_store(self, mock_chroma_client, temp_chroma_dir):
        """Create VectorStore with mocked ChromaDB client"""
        mock_client, mock_collection = mock_chroma_client
        
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_client):
            with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                store = VectorStore(temp_chroma_dir, "test-model", max_results=5)
                store.course_catalog = mock_collection
                store.course_content = mock_collection
                return store, mock_collection
    
    def test_initialization(self, temp_chroma_dir):
        """Test VectorStore initialization"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client_cls:
            with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                mock_client = Mock()
                mock_client_cls.return_value = mock_client
                
                store = VectorStore(temp_chroma_dir, "test-model", max_results=10)
                
                assert store.max_results == 10
                mock_client_cls.assert_called_once()
                mock_client.get_or_create_collection.assert_called()
    
    def test_search_basic_query(self, mock_vector_store):
        """Test basic search without filters"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.return_value = {
            'documents': [['Test document']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        results = store.search("test query")
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
        
        assert len(results.documents) == 1
        assert results.documents[0] == 'Test document'
        assert results.metadata[0]['course_title'] == 'Test Course'
    
    def test_search_with_course_name_resolution(self, mock_vector_store):
        """Test search with course name that needs resolution"""
        store, mock_collection = mock_vector_store
        
        # Mock course name resolution
        store._resolve_course_name = Mock(return_value='Full Course Title')
        
        mock_collection.query.return_value = {
            'documents': [['Test content']],
            'metadatas': [[{'course_title': 'Full Course Title'}]],
            'distances': [[0.1]]
        }
        
        results = store.search("test query", course_name="Partial Course")
        
        store._resolve_course_name.assert_called_once_with("Partial Course")
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Full Course Title"}
        )
    
    def test_search_with_course_name_not_found(self, mock_vector_store):
        """Test search when course name cannot be resolved"""
        store, mock_collection = mock_vector_store
        
        store._resolve_course_name = Mock(return_value=None)
        
        results = store.search("test query", course_name="Unknown Course")
        
        assert results.error == "No course found matching 'Unknown Course'"
        assert results.is_empty()
    
    def test_search_with_lesson_number(self, mock_vector_store):
        """Test search with lesson number filter"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.return_value = {
            'documents': [['Lesson content']],
            'metadatas': [[{'lesson_number': 2}]],
            'distances': [[0.1]]
        }
        
        results = store.search("test query", lesson_number=2)
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"lesson_number": 2}
        )
    
    def test_search_with_both_filters(self, mock_vector_store):
        """Test search with both course name and lesson number"""
        store, mock_collection = mock_vector_store
        
        store._resolve_course_name = Mock(return_value='Test Course')
        
        mock_collection.query.return_value = {
            'documents': [['Specific content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        
        results = store.search("test query", course_name="Test", lesson_number=3)
        
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_with_custom_limit(self, mock_vector_store):
        """Test search with custom result limit"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        store.search("test query", limit=10)
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=10,
            where=None
        )
    
    def test_search_exception_handling(self, mock_vector_store):
        """Test search exception handling"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.side_effect = Exception("Database error")
        
        results = store.search("test query")
        
        assert results.error == "Search error: Database error"
        assert results.is_empty()
    
    def test_resolve_course_name_success(self, mock_vector_store):
        """Test successful course name resolution"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.return_value = {
            'documents': [['Course title']],
            'metadatas': [[{'title': 'Full Course Title'}]]
        }
        
        result = store._resolve_course_name("Partial Title")
        
        mock_collection.query.assert_called_once_with(
            query_texts=["Partial Title"],
            n_results=1
        )
        assert result == "Full Course Title"
    
    def test_resolve_course_name_no_results(self, mock_vector_store):
        """Test course name resolution with no results"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        result = store._resolve_course_name("Unknown Course")
        
        assert result is None
    
    def test_resolve_course_name_exception(self, mock_vector_store):
        """Test course name resolution exception handling"""
        store, mock_collection = mock_vector_store
        
        mock_collection.query.side_effect = Exception("Query failed")
        
        result = store._resolve_course_name("Test Course")
        
        assert result is None
    
    def test_build_filter_no_params(self, mock_vector_store):
        """Test filter building with no parameters"""
        store, _ = mock_vector_store
        
        result = store._build_filter(None, None)
        
        assert result is None
    
    def test_build_filter_course_only(self, mock_vector_store):
        """Test filter building with course title only"""
        store, _ = mock_vector_store
        
        result = store._build_filter("Test Course", None)
        
        assert result == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self, mock_vector_store):
        """Test filter building with lesson number only"""
        store, _ = mock_vector_store
        
        result = store._build_filter(None, 5)
        
        assert result == {"lesson_number": 5}
    
    def test_build_filter_both_params(self, mock_vector_store):
        """Test filter building with both parameters"""
        store, _ = mock_vector_store
        
        result = store._build_filter("Test Course", 3)
        
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        assert result == expected
    
    def test_add_course_metadata(self, mock_vector_store, sample_course):
        """Test adding course metadata"""
        store, mock_collection = mock_vector_store
        
        store.add_course_metadata(sample_course)
        
        # Verify the call was made with correct parameters
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check documents
        assert call_args[1]['documents'] == [sample_course.title]
        
        # Check IDs
        assert call_args[1]['ids'] == [sample_course.title]
        
        # Check metadata structure
        metadata = call_args[1]['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link
        assert metadata['lesson_count'] == len(sample_course.lessons)
        assert 'lessons_json' in metadata
        
        # Verify lessons JSON is valid
        lessons_data = json.loads(metadata['lessons_json'])
        assert len(lessons_data) == len(sample_course.lessons)
    
    def test_add_course_content(self, mock_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        store, mock_collection = mock_vector_store
        
        store.add_course_content(sample_course_chunks)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check documents
        expected_docs = [chunk.content for chunk in sample_course_chunks]
        assert call_args[1]['documents'] == expected_docs
        
        # Check metadata
        expected_metadata = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in sample_course_chunks]
        assert call_args[1]['metadatas'] == expected_metadata
        
        # Check IDs
        expected_ids = [f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}" 
                       for chunk in sample_course_chunks]
        assert call_args[1]['ids'] == expected_ids
    
    def test_add_course_content_empty_list(self, mock_vector_store):
        """Test adding empty course content list"""
        store, mock_collection = mock_vector_store
        
        store.add_course_content([])
        
        # Should not call add if list is empty
        mock_collection.add.assert_not_called()
    
    def test_clear_all_data(self, mock_vector_store):
        """Test clearing all data"""
        store, _ = mock_vector_store
        store.client = Mock()
        
        store.clear_all_data()
        
        # Should delete both collections
        assert store.client.delete_collection.call_count == 2
        store.client.delete_collection.assert_any_call("course_catalog")
        store.client.delete_collection.assert_any_call("course_content")
    
    def test_clear_all_data_exception(self, mock_vector_store):
        """Test clear all data with exception"""
        store, _ = mock_vector_store
        store.client = Mock()
        store.client.delete_collection.side_effect = Exception("Delete failed")
        
        # Should not raise exception
        store.clear_all_data()
    
    def test_get_existing_course_titles(self, mock_vector_store):
        """Test getting existing course titles"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.return_value = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        
        titles = store.get_existing_course_titles()
        
        assert titles == ['Course A', 'Course B', 'Course C']
        mock_collection.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self, mock_vector_store):
        """Test getting course titles when none exist"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.return_value = {'ids': []}
        
        titles = store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_existing_course_titles_exception(self, mock_vector_store):
        """Test getting course titles with exception"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.side_effect = Exception("Get failed")
        
        titles = store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_course_count(self, mock_vector_store):
        """Test getting course count"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.return_value = {
            'ids': ['Course A', 'Course B']
        }
        
        count = store.get_course_count()
        
        assert count == 2
    
    def test_get_course_count_exception(self, mock_vector_store):
        """Test getting course count with exception"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.side_effect = Exception("Count failed")
        
        count = store.get_course_count()
        
        assert count == 0
    
    def test_get_all_courses_metadata(self, mock_vector_store):
        """Test getting all courses metadata"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "link1"}
        ])
        
        mock_collection.get.return_value = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Test Instructor',
                'lessons_json': lessons_json,
                'lesson_count': 1
            }]
        }
        
        metadata = store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        assert metadata[0]['title'] == 'Test Course'
        assert metadata[0]['instructor'] == 'Test Instructor'
        assert 'lessons' in metadata[0]
        assert 'lessons_json' not in metadata[0]  # Should be removed
        assert len(metadata[0]['lessons']) == 1
    
    def test_get_course_link(self, mock_vector_store):
        """Test getting course link"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.return_value = {
            'metadatas': [{'course_link': 'https://example.com/course'}]
        }
        
        link = store.get_course_link("Test Course")
        
        assert link == 'https://example.com/course'
        mock_collection.get.assert_called_once_with(ids=["Test Course"])
    
    def test_get_course_link_not_found(self, mock_vector_store):
        """Test getting course link when course not found"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        mock_collection.get.return_value = {'metadatas': []}
        
        link = store.get_course_link("Unknown Course")
        
        assert link is None
    
    def test_get_lesson_link(self, mock_vector_store):
        """Test getting lesson link"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_link": "https://example.com/lesson2"}
        ])
        
        mock_collection.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        
        link = store.get_lesson_link("Test Course", 2)
        
        assert link == 'https://example.com/lesson2'
    
    def test_get_lesson_link_not_found(self, mock_vector_store):
        """Test getting lesson link when lesson not found"""
        store, mock_collection = mock_vector_store
        store.course_catalog = mock_collection
        
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}
        ])
        
        mock_collection.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        
        link = store.get_lesson_link("Test Course", 5)  # Non-existent lesson
        
        assert link is None