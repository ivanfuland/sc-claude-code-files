from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Maximum 2 sequential searches per query** - You can search, analyze results, then search again if needed
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Multi-step Reasoning:
- For complex queries, you may need multiple searches to gather complete information
- Example: "Search for course X outline" → analyze lesson 4 topic → "Search for courses covering that topic"
- Each search builds upon previous results to provide comprehensive answers

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex queries**: Use multiple searches as needed (max 2)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        _round: int = 0,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            _round: Internal parameter for tracking tool call rounds (max 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(
                response, api_params, tool_manager, _round
            )

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        current_round: int = 0,
    ):
        """
        Handle execution of tool calls and get follow-up response.
        Supports up to 2 sequential rounds of tool calling.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            current_round: Current round number (0 or 1)

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        try:
            for content_block in initial_response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
        except Exception as e:
            # Handle tool execution errors gracefully
            if current_round == 0:
                # First round: re-raise the exception
                raise e
            else:
                # Second round: return friendly error message
                return "I encountered an issue while searching for additional information. Please try rephrasing your question."

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Check if we can make another round (max 2 rounds total)
        if current_round < 1:
            # Prepare API call with tools still available for potential second round
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools"),
                "tool_choice": base_params.get("tool_choice"),
            }

            # Get response from Claude (might use tools again)
            next_response = self.client.messages.create(**next_params)

            # If Claude wants to use tools again, handle recursively
            if next_response.stop_reason == "tool_use":
                # Create new base_params for the recursive call
                recursive_base_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": base_params.get("tools"),
                    "tool_choice": base_params.get("tool_choice"),
                }

                return self._handle_tool_execution(
                    next_response,
                    recursive_base_params,
                    tool_manager,
                    current_round + 1,
                )
            else:
                # Claude doesn't want to use tools again, return response
                return next_response.content[0].text
        else:
            # Maximum rounds reached, make final call without tools
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }

            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
