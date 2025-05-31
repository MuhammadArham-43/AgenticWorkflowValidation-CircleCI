import asyncio
from agent.tools import get_coordinates_from_city, get_current_weather, search_wikipedia, calculate
from agent.agent import LLMAgent

async def main():
    tools = [get_coordinates_from_city, get_current_weather, search_wikipedia, calculate]
    agent = LLMAgent(tools=tools, model_name="gpt-3.5-turbo")
    print("--- Testing LLM Agent ---")

    # Test Case 1: Weather query (multi-tool use: geocoding -> weather)
    response = await agent.run_query("What's the current weather like in Karachi?")
    print(f"--- Agent Final Response ---\n{response}")


if __name__ == "__main__":
    asyncio.run(main())    