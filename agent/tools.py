import typing as T
import json
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError


class WikipediaArticle(BaseModel):
    """Represents a summary of a Wikipedia article"""
    title: str = Field(description="Title of the Wikipedia article")
    summary: str = Field(description="Summary of the Wikipedia article")
    url: str = Field(description="URL of the Wikipedia article")

class Coordinates(BaseModel):
    """Represents geographical coordinates for a location"""
    latitude: float = Field(description="Latitude of the location")
    longitude: float = Field(description="Longitude of the location")

class CurrentWeather(BaseModel):
    """Represents the current weather conditions for a location"""
    latitude: float = Field(description="Latitude of the location")
    longitude: float = Field(description="Longitude of the location")
    temperature: float = Field(description="Current temperature in degrees Celsius")
    wind_speed: float = Field(description="Current wind speed in km/h")
    relative_humidity_2m: float = Field(description="Current relative humidity at 2 meters above ground level in percentage")
    is_day: int = Field(description="Indicates if it is currently day (1) or night (0) at the location")
    weather_code: int = Field(description="WMO Weather interpretation code")
    time: str = Field(description="Current time of the weather observation in ISO format")


@tool
def get_coordinates_from_city(city_name: str) -> str:
    """
    Converts a city name into a geographical latitutde and longitude using Open-Meteo Geocoding API.
    Returns a JSON string of a coordinate object for the most relevant result.
    """
    GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        response = requests.get(GEOCODING_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results")
        if not results:
            return json.dumps({"error": f"Could not find coordinates for city: {city_name}"})

        top_result = results[0]
        results_data = {
            "name": top_result.get("name"),
            "latitude": top_result.get("latitude"),
            "longitude": top_result.get("longitude"),
            "country": top_result.get("country"),
        }
        validated_result = Coordinates(**results_data)
        return validated_result.model_dump_json()
    
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Geocoding API request timed out."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error connecting to Geocoding API: {e}"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to decode JSON response from Geocoding API."})
    except ValidationError as e:
        return json.dumps({"error": f"Failed to validate coordinates schema: {e.errors()}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during coordinate retrieval: {e}"})


@tool
def get_current_weather(latitude: float, longitude: float) -> str:
    """
    Retrieves the current weather conditions for a given latitude and longitude using Open-Meteo Weather API.
    Returns a JSON string of a weather object.
    """
    WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,is_day,wind_speed_10m,weather_code",
        "timezone": "auto",
        "forecast_days": 1
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            return json.dumps({"error": "No current weather data available for this location."})

        current_data = data["current"]
        weather_data = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": current_data.get("temperature_2m"),
            "wind_speed": current_data.get("wind_speed_10m"),
            "relative_humidity_2m": current_data.get("relative_humidity_2m"),
            "is_day": current_data.get("is_day"),
            "weather_code": current_data.get("weather_code"),
            "time": current_data.get("time")
        }
        validated_weather = CurrentWeather(**weather_data) 
        return validated_weather.model_dump_json()

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Weather API request timed out."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error connecting to Weather API: {e}"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to decode JSON response from Weather API."})
    except ValidationError as e:
        return json.dumps({"error": f"Failed to validate weather data schema: {e.errors()}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during weather retrieval: {e}"})


@tool
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for a given query and returns a summary and URL.
    Returns a JSON string of a Wikipedia object.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": query,
        "prop": "extracts|info",
        "exintro": True, # Get only introductory section
        "explaintext": True,
        "inprop": "url",
        "redirects": 1
    }

    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return json.dumps({"error": "No wikipedia article found for the query"})

        page_id = next(iter(pages))
        page_data = pages[page_id]

        if "missing" in page_data:
            return json.dumps({"error": "No wikipedia article found the query"})
        
        results_data = {
            "title": page_data.get("title"),
            "summary": page_data.get("summary") if "summary" in page_data else page_data.get("extract"),
            "url": page_data.get("fullurl")
        }
        validated_result = WikipediaArticle(**results_data)
        return validated_result.model_dump_json()

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Wikipedia API request timed out."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error connecting to Wikipedia API: {e}"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to decode JSON response from Wikipedia API."})
    except ValidationError as e:
        return json.dumps({"error": f"Failed to validate Wikipedia article schema: {e.errors()}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during Wikipedia search: {e}"})


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression (e.g '2 + 2 * 3').
    Supports basic arithmetic operations
    """
    try:
        result = eval(expression)
        return str(result)
    
    except SyntaxError:
        return "Error: Invalid mathematical expression."
    except NameError:
        return "Error: Invalid input in expression (e.g., non-numeric characters)."
    except Exception as e:
        return f"Error during calculation: {e}"