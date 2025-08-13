"""
Unit tests for AIGenerator class.

Tests the AI response generation, tool calling mechanism, and API integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""
    
    @pytest.fixture
    def mock_anthropic_response_text(self):
        """Create a mock Anthropic response for text generation"""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "This is a test response from Claude."
        mock_response.content = [mock_content]
        return mock_response
    
    @pytest.fixture
    def mock_anthropic_response_tool_use(self):
        """Create a mock Anthropic response for tool use"""
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_call_123"
        mock_tool_content.input = {"query": "test query", "course_name": "Python"}
        mock_response.content = [mock_tool_content]
        return mock_response
    
    @pytest.fixture
    def mock_anthropic_client(self, mock_anthropic_response_text):
        """Create a mock Anthropic client"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text
        return mock_client
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        manager = Mock()
        manager.execute_tool.return_value = "Search results from tool"
        return manager
    
    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-api-key", "claude-test-model")
            
            assert generator.model == "claude-test-model"
            assert generator.base_params["model"] == "claude-test-model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
            
            mock_anthropic.assert_called_once_with(api_key="test-api-key")
    
    def test_system_prompt_constant(self):
        """Test that system prompt is properly defined"""
        assert hasattr(AIGenerator, 'SYSTEM_PROMPT')
        assert isinstance(AIGenerator.SYSTEM_PROMPT, str)
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
        assert "search tool" in AIGenerator.SYSTEM_PROMPT.lower()
    
    def test_generate_response_text_only(self, mock_anthropic_client, mock_anthropic_response_text):
        """Test generating response without tools"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            response = generator.generate_response("What is Python?")
            
            assert response == "This is a test response from Claude."
            
            # Verify API call
            mock_anthropic_client.messages.create.assert_called_once()
            call_args = mock_anthropic_client.messages.create.call_args[1]
            
            assert call_args["model"] == "test-model"
            assert call_args["temperature"] == 0
            assert call_args["max_tokens"] == 800
            assert call_args["messages"] == [{"role": "user", "content": "What is Python?"}]
            assert call_args["system"] == AIGenerator.SYSTEM_PROMPT
            assert "tools" not in call_args
    
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test generating response with conversation history"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            history = "Previous conversation context"
            generator.generate_response("Follow up question", conversation_history=history)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            expected_system = f"{AIGenerator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{history}"
            assert call_args["system"] == expected_system
    
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, sample_tools, mock_tool_manager):
        """Test response generation with tools available but not used"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            response = generator.generate_response(
                "What is 2+2?",
                tools=sample_tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert call_args["tools"] == sample_tools
            assert call_args["tool_choice"] == {"type": "auto"}
            
            # Tool manager should not be called
            mock_tool_manager.execute_tool.assert_not_called()
    
    def test_generate_response_with_tool_use(self, mock_anthropic_client, mock_anthropic_response_tool_use, sample_tools, mock_tool_manager):
        """Test response generation with tool use"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            # First call returns tool use, second call returns final response
            final_response = Mock()
            final_response.content = [Mock(text="Final answer with tool results")]
            
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_response_tool_use,
                final_response
            ]
            
            response = generator.generate_response(
                "Search for Python courses",
                tools=sample_tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "Final answer with tool results"
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="Python"
            )
            
            # Verify two API calls were made
            assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, mock_anthropic_client, mock_anthropic_response_tool_use, mock_tool_manager):
        """Test handling of single tool execution"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system"
            }
            
            # Mock final response
            final_response = Mock()
            final_response.content = [Mock(text="Tool response integrated")]
            mock_anthropic_client.messages.create.return_value = final_response
            
            result = generator._handle_tool_execution(
                mock_anthropic_response_tool_use,
                base_params,
                mock_tool_manager
            )
            
            assert result == "Tool response integrated"
            
            # Verify tool execution
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="Python"
            )
            
            # Verify final API call structure
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert len(call_args["messages"]) == 3  # Original + assistant + tool result
            assert call_args["messages"][1]["role"] == "assistant"
            assert call_args["messages"][2]["role"] == "user"
            assert "tools" not in call_args  # Tools should not be included in final call
    
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client, mock_tool_manager):
        """Test handling multiple tool executions in one response"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            # Create response with multiple tool uses
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            
            tool1 = Mock()
            tool1.type = "tool_use"
            tool1.name = "search_course_content"
            tool1.id = "tool1_id"
            tool1.input = {"query": "first query"}
            
            tool2 = Mock()
            tool2.type = "tool_use"
            tool2.name = "search_course_content"
            tool2.id = "tool2_id"
            tool2.input = {"query": "second query"}
            
            mock_response.content = [tool1, tool2]
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "system": "test system"
            }
            
            # Configure tool manager to return different results
            mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
            
            # Mock final response
            final_response = Mock()
            final_response.content = [Mock(text="Multiple tools handled")]
            mock_anthropic_client.messages.create.return_value = final_response
            
            result = generator._handle_tool_execution(mock_response, base_params, mock_tool_manager)
            
            assert result == "Multiple tools handled"
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="second query")
    
    def test_handle_tool_execution_no_tool_results(self, mock_anthropic_client):
        """Test tool execution handling when no tools are found"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            # Create response with no tool use content
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            text_content = Mock()
            text_content.type = "text"
            mock_response.content = [text_content]
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "system": "test system"
            }
            
            mock_tool_manager = Mock()
            
            # Mock final response
            final_response = Mock()
            final_response.content = [Mock(text="No tools executed")]
            mock_anthropic_client.messages.create.return_value = final_response
            
            result = generator._handle_tool_execution(mock_response, base_params, mock_tool_manager)
            
            assert result == "No tools executed"
            
            # No tools should be executed
            mock_tool_manager.execute_tool.assert_not_called()
    
    def test_api_parameter_construction(self, mock_anthropic_client, sample_tools):
        """Test that API parameters are correctly constructed"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            generator.generate_response(
                "Test query",
                conversation_history="Previous chat",
                tools=sample_tools
            )
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            
            # Verify all expected parameters are present
            assert "model" in call_args
            assert "temperature" in call_args
            assert "max_tokens" in call_args
            assert "messages" in call_args
            assert "system" in call_args
            assert "tools" in call_args
            assert "tool_choice" in call_args
            
            # Verify parameter values
            assert call_args["model"] == "test-model"
            assert call_args["temperature"] == 0
            assert call_args["max_tokens"] == 800
            assert call_args["tools"] == sample_tools
            assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_error_handling_api_exception(self, mock_anthropic_client):
        """Test error handling when API raises exception"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            mock_anthropic_client.messages.create.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                generator.generate_response("Test query")
    
    def test_error_handling_tool_manager_exception(self, mock_anthropic_client, mock_anthropic_response_tool_use):
        """Test error handling when tool manager raises exception"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
            
            # Should let the exception propagate or handle gracefully
            with pytest.raises(Exception):
                generator._handle_tool_execution(
                    mock_anthropic_response_tool_use,
                    {"messages": [], "system": "test"},
                    mock_tool_manager
                )
    
    def test_response_content_access(self, mock_anthropic_client):
        """Test accessing response content safely"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            # Test with malformed response
            malformed_response = Mock()
            malformed_response.stop_reason = "end_turn"
            malformed_response.content = []  # Empty content
            
            mock_anthropic_client.messages.create.return_value = malformed_response
            
            # Should raise an IndexError for empty content
            with pytest.raises(IndexError):
                generator.generate_response("Test query")
    
    def test_system_prompt_with_history_construction(self, mock_anthropic_client):
        """Test system prompt construction with conversation history"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            history = "User: Hello\nAssistant: Hi there!"
            generator.generate_response("Follow up", conversation_history=history)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            system_content = call_args["system"]
            
            assert AIGenerator.SYSTEM_PROMPT in system_content
            assert "Previous conversation:" in system_content
            assert history in system_content
    
    def test_tool_choice_parameter(self, mock_anthropic_client, sample_tools):
        """Test tool_choice parameter is set correctly"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            generator.generate_response("Test query", tools=sample_tools)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_base_params_optimization(self):
        """Test that base parameters are pre-built for efficiency"""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator("test-key", "test-model")
            
            assert hasattr(generator, 'base_params')
            assert generator.base_params["model"] == "test-model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_message_construction(self, mock_anthropic_client):
        """Test message list construction"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            query = "What is machine learning?"
            generator.generate_response(query)
            
            call_args = mock_anthropic_client.messages.create.call_args[1]
            messages = call_args["messages"]
            
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == query
    
    def test_final_api_call_without_tools(self, mock_anthropic_client, mock_anthropic_response_tool_use, mock_tool_manager):
        """Test that final API call removes tools parameter"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "system": "test system",
                "tools": [{"name": "test_tool"}],  # Should be removed in final call
                "tool_choice": {"type": "auto"}   # Should be removed in final call
            }
            
            final_response = Mock()
            final_response.content = [Mock(text="Final response")]
            mock_anthropic_client.messages.create.return_value = final_response
            
            generator._handle_tool_execution(mock_anthropic_response_tool_use, base_params, mock_tool_manager)
            
            # Verify final call doesn't include tools
            final_call_args = mock_anthropic_client.messages.create.call_args[1]
            assert "tools" not in final_call_args
            assert "tool_choice" not in final_call_args