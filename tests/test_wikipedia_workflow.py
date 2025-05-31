import pytest
import json
from unittest.mock import patch

from agent.tools import WikipediaArticle
from .conftest import get_agent_trajectory, create_mock_response



@pytest.mark.asyncio
async def test_wikipedia_search_success_workflow(llm_agent):
    """
    Tests the workflow for a successful Wikipedia search query.
    1. Agent calls search_wikipedia.
    2. Agent provides a final response using Wikipedia summary.
    3. Validates structured JSON output from the tool.
    """

    mock_response_wiki = create_mock_response(
        status_code=200,
        json_data={
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "ns": 0,
                        "title": "Artificial intelligence",
                        "extract": "Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines...",
                        "fullurl": "https://en.wikipedia.org/wiki/Artificial_intelligence"
                    }
                }
            }
        }
    )

    with patch('agent.tools.requests.get', return_value=mock_response_wiki) as mock_get_func: # Use mock_get_func
        query = "Tell me about artificial intelligence."
        trajectory = await get_agent_trajectory(llm_agent, query)

        # Validate tool call
        wiki_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "search_wikipedia"), None)
        assert wiki_call is not None, "Agent did not call search_wikipedia"
        assert wiki_call["tool_args"]["query"].lower() == "artificial intelligence", "search_wikipedia called with incorrect query"

        # Validate Wikipedia tool output and its structure
        wiki_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "search_wikipedia"), None)
        assert wiki_output is not None, "Missing output from search_wikipedia"
        parsed_wiki = json.loads(wiki_output["tool_output"])
        validated_wiki = WikipediaArticle(**parsed_wiki) # Validate against Pydantic model
        assert validated_wiki.title.lower() == "artificial intelligence".lower()
        assert "intelligence—perceiving" in validated_wiki.summary
        assert "https://en.wikipedia.org/wiki/Artificial_intelligence" == validated_wiki.url

        # Validate final agent response
        final_response = next((s for s in trajectory if s["type"] == "final_response"), None)
        assert final_response is not None, "Missing final response from agent"
        assert "Artificial intelligence" in final_response["content"]
        assert "machines" in final_response["content"]


@pytest.mark.asyncio
# Removed @patch decorator, will use with patch inside the test
async def test_wikipedia_search_not_found_error_handling(llm_agent): # Removed 'mock_get' from arguments
    """
    Tests agent's error handling when Wikipedia API finds no article.
    """

    mock_response_no_results = create_mock_response(
        status_code=200,
        json_data={"query": {"pages": {"-1": {"missing": ""}}}} # Wikipedia API response for no article found
    )

    with patch('agent.tools.requests.get', return_value=mock_response_no_results) as mock_get_func: # Use mock_get_func
        query = "Summarize the life of Zorp the Destroyer."
        trajectory = await get_agent_trajectory(llm_agent, query)

        # Validate search_wikipedia call
        wiki_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "search_wikipedia"), None)
        assert wiki_call is not None, "Agent did not call search_wikipedia"
        assert wiki_call["tool_args"]["query"] == "Zorp the Destroyer"

        # Validate tool output indicates error
        wiki_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "search_wikipedia"), None)
        assert wiki_output is not None, "Missing output from search_wikipedia"
        parsed_output = json.loads(wiki_output["tool_output"])
        assert "error" in parsed_output
        assert "No Wikipedia article found".lower() in parsed_output["error"].lower()

        # Validate final agent response reflects the error
        final_response = next((s for s in trajectory if s["type"] == "final_response"), None)
        assert final_response is not None
        assert "could not find" in final_response["content"].lower() or "no information" in final_response["content"].lower() or "couldn't find" in final_response["content"].lower()
        assert "zorp the destroyer" in final_response["content"].lower()