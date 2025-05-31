import typing as T
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode



class AgentState(T.TypedDict):
    messages: T.Annotated[T.List[BaseMessage], add_messages]

class LLMAgent:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        tools: T.List[T.Callable[..., T.Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> None:
        """
        Initializes the LLM agent with tools and an OpenAI LLM.
        Args:
            model_name (str): The name of the OpenAI model to use
            tools (list): A list of LangChain tools the agent can use.
            max_tokens (int): The max tokens to genreate in the response.
            temperature (float): The sampling temperature for the LLM.
        """
        self.tools = tools or []
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.app = self._build_workflow()
    
    def _call_model(self, state: AgentState):
        """
        Internal node method that calls the LLM with the current message history.
        The LLM decides whether to respond directly or call a tool.
        """
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """
        Internal method that determines whether the agent should continue by calling a tool
        or end the conversation. This logic inspects the last message from the LLM.
        """
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def _build_workflow(self):
        """Builds and compiles the LangGraph workflow for the agent."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent", 
            self._should_continue,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    async def run_query(self, query: str) -> str:
        """
        Runs a query through the agent's workflow.
        Args:
            query (str): The input query for the agent.
        Returns:
            str: The final response from the agent.
        """
        inputs = {"messages": [HumanMessage(content=query)]}
        full_response = []
        async for s in self.app.astream(inputs):
            if "__end__" not in s:
                full_response.append(s)
       
        final_state = full_response[-1]
        final_message = final_state["agent"]["messages"][-1]
        return final_message.content

    async def stream_query(self, query: str):
        """
        Streams the intermediate steps and final response of a query.
        Useful for debugging and observing agent behavior.

        Args:
            query (str): The user's input query.

        Yields:
            Dict: The intermediate state or final response.
        """
        inputs = {"messages": [HumanMessage(content=query)]}
        async for s in self.app.astream(inputs):
            yield s