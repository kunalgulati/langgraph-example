from typing import TypedDict, Literal, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GraphConfig(TypedDict):
    """The configuration for the graph."""
    model_name: Literal["openai"]
