from typing import Dict
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage

from agent.agent import LLMAgent
from agent.tools import calculate, get_current_weather, get_coordinates_from_city, search_wikipedia


@pytest.fixture(scope="module")
def llm_agent():
    """
    Provides an instance of LLMAgent for testing
    We use a module scope to initialize it once for all tests in this file
    """
    tools = [calculate, get_coordinates_from_city, get_current_weather, search_wikipedia]
    return LLMAgent(tools=tools, model_name="gpt-3.5-turbo", temperature=0.1)


async def get_agent_trajectory(agent: LLMAgent, query: str):
    """
    Runs the agent in streaming mode and extracts relevant steps for validation.
    Returns a list of dictionaries, each representing a significant event.
    """
    trajectory = []
    async for step in agent.stream_query(query):
        # LangGraph stream yields dictionaries representing state changes or events
        print(step)
        if "agent" not in step and "tools" not in step: continue
        last_message = step["agent"]["messages"][-1] if "agent" in step else step["tools"]["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # LLM decided to call a tool
            for tool_call in last_message.tool_calls:
                trajectory.append({
                    "type": "tool_call",
                    "tool_name": tool_call['name'],
                    "tool_args": tool_call['args']
                })
        elif isinstance(last_message, ToolMessage):
            # Tool execution result
            trajectory.append({
                "type": "tool_output",
                "tool_name": last_message.name,
                "tool_output": last_message.content # This is the JSON string
            })
        elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
            # Final LLM response
            trajectory.append({
                "type": "final_response",
                "content": last_message.content
            })
    return trajectory


def create_mock_response(status_code: int, json_data: Dict = None, text_data: str = None):
    """
    Helper to create a MagicMock object that simulates a requests.Response.
    """
    mock_response = MagicMock(status_code=status_code)
    if json_data is not None:
        mock_response.json.return_value = json_data
    elif text_data is not None:
        mock_response.text = text_data # For APIs that might return plain text
    mock_response.raise_for_status.return_value = None # Assume success unless status_code indicates error
    return mock_response