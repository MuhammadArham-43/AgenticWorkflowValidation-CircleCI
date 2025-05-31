import pytest
from .conftest import get_agent_trajectory

@pytest.mark.asyncio
async def test_calculator_success_workflow(llm_agent):
    """
    Tests the workflow for a successful calculation query.
    1. Agent calls calculate.
    2. Agent provides a final response with the calculation result.
    """
    query = "What is 10 + 5 * 2?"
    trajectory = await get_agent_trajectory(llm_agent, query) # Changed from pytest.helpers.get_agent_trajectory

    # Validate tool call
    calc_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "calculate"), None)
    assert calc_call is not None, "Agent did not call calculate"
    assert calc_call["tool_args"]["expression"] == "10 + 5 * 2", "calculate called with incorrect expression"

    # Validate Calculator tool output
    calc_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "calculate"), None)
    assert calc_output is not None, "Missing output from calculate"
    assert calc_output["tool_output"] == "20", "Calculator returned incorrect result" # 10 + (5*2) = 20

    # Validate final agent response
    final_response = next((s for s in trajectory if s["type"] == "final_response"), None)
    assert final_response is not None, "Missing final response from agent"
    assert "20" in final_response["content"], "Final response does not contain calculation result"