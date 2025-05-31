import pytest
import json
from unittest.mock import patch

from agent.tools import Coordinates, CurrentWeather
from .conftest import get_agent_trajectory, create_mock_response


@pytest.mark.asyncio
async def test_weather_query_success_workflow(llm_agent):
    """
    Tests the complete workflow for a weather query:
    1. Agent calls get_coordinates_from_city.
    2. Agent calls get_current_weather with coordinates from step 1.
    3. Agent provides a final response using weather data.
    4. Validates structured JSON outputs from tools.
    """
    # Mock responses for the external APIs
    # Mock 1: Geocoding API response for "London"
    mock_response_geo = create_mock_response(
        status_code=200,
        json_data={
            "results": [{
                "name": "London",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "country": "United Kingdom",
                "admin1": "England"
            }]
        }
    )

    # Mock 2: Open-Meteo Weather API response
    mock_response_weather = create_mock_response(
        status_code=200,
        json_data={
            "latitude": 51.5074,
            "longitude": -0.1278,
            "current": {
                "temperature_2m": 15.5,
                "wind_speed_10m": 12.3,
                "relative_humidity_2m": 75,
                "is_day": 1,
                "weather_code": 3, # Example WMO code for 'cloudy'
                "time": "2025-06-01T10:00Z"
            }
        }
    )
    
    with patch('agent.tools.requests.get', side_effect=[mock_response_geo, mock_response_weather]) as mock_get_func: # Use mock_get_func
        query = "What's the weather like in London?"
        trajectory = await get_agent_trajectory(llm_agent, query)
        
        assert len(trajectory) >= 3, "Expected at least 3 steps: geocoding call, weather call, final response"

        # Step 1: Validate get_coordinates_from_city call
        coord_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "get_coordinates_from_city"), None)
        assert coord_call is not None, "Agent did not call get_coordinates_from_city"
        assert coord_call["tool_args"]["city_name"] == "London", "get_coordinates_from_city called with incorrect city name"

        # Step 2: Validate Coordinates tool output and its structure
        coord_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "get_coordinates_from_city"), None)
        assert coord_output is not None, "Missing output from get_coordinates_from_city"
        parsed_coords = json.loads(coord_output["tool_output"])
        validated_coords = Coordinates(**parsed_coords) # Validate against Pydantic model
        assert validated_coords.latitude == pytest.approx(51.5074)
        assert validated_coords.longitude == pytest.approx(-0.1278)

        # Step 3: Validate get_current_weather call (using output from geocoding)
        weather_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "get_current_weather"), None)
        assert weather_call is not None, "Agent did not call get_current_weather"
        assert weather_call["tool_args"]["latitude"] == pytest.approx(51.5074) # Use pytest.approx for floats
        assert weather_call["tool_args"]["longitude"] == pytest.approx(-0.1278)

        # Step 4: Validate CurrentWeather tool output and its structure
        weather_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "get_current_weather"), None)
        assert weather_output is not None, "Missing output from get_current_weather"
        parsed_weather = json.loads(weather_output["tool_output"])
        validated_weather = CurrentWeather(**parsed_weather) # Validate against Pydantic model
        assert validated_weather.temperature == 15.5
        assert validated_weather.wind_speed == 12.3
        assert validated_weather.relative_humidity_2m == 75

        # Step 5: Validate final agent response
        final_response = next((s for s in trajectory if s["type"] == "final_response"), None)
        assert final_response is not None, "Missing final response from agent"
        assert "15.5" in final_response["content"] # Check for temperature in response
        assert "London" in final_response["content"] # Check for city name in response
        # The LLM's interpretation of weather code 3 (cloudy) might vary, so a flexible check
        assert any(keyword in final_response["content"].lower() for keyword in ["cloudy", "partly cloudy", "overcast"])

@pytest.mark.asyncio
async def test_weather_city_not_found_error_handling(llm_agent):
    """
    Tests agent's error handling when geocoding API cannot find a city.
    """
    # Mock 1: Geocoding API response for "imaginary_city_123" (no results)
    mock_response_no_results = create_mock_response(
        status_code=200,
        json_data={"results": []} # No results found
    )
    with patch('agent.tools.requests.get', return_value=mock_response_no_results) as mock_get_func: # Use mock_get_func
        query = "What's the weather in imaginary_city_123?"
        trajectory = await get_agent_trajectory(llm_agent, query)

        # Validate get_coordinates_from_city call
        coord_call = next((s for s in trajectory if s["type"] == "tool_call" and s["tool_name"] == "get_coordinates_from_city"), None)
        assert coord_call is not None, "Agent did not call get_coordinates_from_city"
        assert coord_call["tool_args"]["city_name"] == "imaginary_city_123"

        # Validate tool output indicates error
        coord_output = next((s for s in trajectory if s["type"] == "tool_output" and s["tool_name"] == "get_coordinates_from_city"), None)
        assert coord_output is not None, "Missing output from get_coordinates_from_city"
        parsed_output = json.loads(coord_output["tool_output"])
        assert "error" in parsed_output
        assert "Could not find coordinates" in parsed_output["error"]

        # Validate final agent response reflects the error
        final_response = next((s for s in trajectory if s["type"] == "final_response"), None)
        assert final_response is not None
        assert "could not find" in final_response["content"].lower() or "unable to determine" in final_response["content"].lower() or "couldn't find" in final_response["content"].lower() 
        assert "imaginary_city_123" in final_response["content"]
