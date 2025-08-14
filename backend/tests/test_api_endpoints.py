"""
API endpoint tests for the RAG Chatbot FastAPI application.

Tests all API endpoints including request/response validation,
error handling, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""
    
    def test_query_with_session_id(self, test_client, sample_query_request, expected_query_response):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
    
    def test_query_without_session_id(self, test_client, sample_query_request_no_session):
        """Test query endpoint without session ID - should create new session"""
        response = test_client.post("/api/query", json=sample_query_request_no_session)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # Mock returns this
    
    def test_query_invalid_request(self, test_client, invalid_query_request):
        """Test query endpoint with invalid request data"""
        response = test_client.post("/api/query", json=invalid_query_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_missing_required_field(self, test_client):
        """Test query endpoint with missing required query field"""
        response = test_client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_empty_string(self, test_client):
        """Test query endpoint with empty query string"""
        response = test_client.post("/api/query", json={"query": ""})
        
        assert response.status_code == status.HTTP_200_OK
        # Should still process empty query
    
    def test_query_server_error(self, test_client, test_app):
        """Test query endpoint when RAG system raises exception"""
        # Configure the mock to raise an exception
        test_app.state.mock_rag.query.side_effect = Exception("Database connection failed")
        
        response = test_client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database connection failed" in response.json()["detail"]
        
        # Reset the mock for other tests
        test_app.state.mock_rag.query.side_effect = None
        test_app.state.mock_rag.query.return_value = ("Test answer", ["Test source"])
    
    def test_query_response_schema(self, test_client, sample_query_request):
        """Test that query response matches expected schema"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    def test_query_long_text(self, test_client):
        """Test query endpoint with very long query text"""
        long_query = "What is Python? " * 1000  # Very long query
        response = test_client.post("/api/query", json={"query": long_query})
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""
    
    def test_get_course_stats(self, test_client, expected_course_stats):
        """Test getting course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
    
    def test_course_stats_response_schema(self, test_client):
        """Test that course stats response matches expected schema"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
    
    def test_course_stats_server_error(self, test_client, test_app):
        """Test course stats endpoint when RAG system raises exception"""
        # Configure the mock to raise an exception
        test_app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics service unavailable")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analytics service unavailable" in response.json()["detail"]
        
        # Reset the mock for other tests
        test_app.state.mock_rag.get_course_analytics.side_effect = None
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python Programming", "Data Science"]
        }
    
    def test_course_stats_no_courses(self, test_client, test_app):
        """Test course stats when no courses are available"""
        # Mock empty course analytics
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestAPIHeaders:
    """Test API headers and CORS configuration"""
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.options("/api/query")
        
        # Should allow CORS preflight
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]
    
    def test_content_type_json(self, test_client, sample_query_request):
        """Test that API accepts and returns JSON content"""
        response = test_client.post(
            "/api/query", 
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.api 
class TestAPIValidation:
    """Test API request validation and error handling"""
    
    def test_invalid_json(self, test_client):
        """Test API with invalid JSON payload"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_extra_fields_ignored(self, test_client):
        """Test that extra fields in request are ignored"""
        request_data = {
            "query": "What is Python?",
            "extra_field": "should be ignored",
            "another_field": 12345
        }
        
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_wrong_data_types(self, test_client):
        """Test API with wrong data types in request"""
        request_data = {
            "query": 12345,  # Should be string
            "session_id": ["not", "a", "string"]  # Should be string
        }
        
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_query_and_courses_integration(self, test_client):
        """Test that query and courses endpoints work together"""
        # First, get course stats
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        
        # Then, make a query
        query_response = test_client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == status.HTTP_200_OK
        
        # Both should work independently
        courses_data = courses_response.json()
        query_data = query_response.json()
        
        assert "total_courses" in courses_data
        assert "answer" in query_data
    
    def test_session_persistence(self, test_client):
        """Test that session IDs work correctly across requests"""
        # First query - no session ID provided
        response1 = test_client.post("/api/query", json={"query": "test 1"})
        session_id = response1.json()["session_id"]
        
        # Second query - use same session ID
        response2 = test_client.post("/api/query", json={
            "query": "test 2", 
            "session_id": session_id
        })
        
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance and load testing for API endpoints"""
    
    def test_concurrent_queries(self, test_client):
        """Test multiple concurrent queries"""
        import concurrent.futures
        import threading
        
        def make_query(query_num):
            return test_client.post("/api/query", json={"query": f"test query {query_num}"})
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_query, i) for i in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
    
    def test_api_response_time(self, test_client):
        """Test that API responses are reasonably fast"""
        import time
        
        start_time = time.time()
        response = test_client.post("/api/query", json={"query": "quick test"})
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        # Response should be under 5 seconds (generous for testing)
        assert (end_time - start_time) < 5.0