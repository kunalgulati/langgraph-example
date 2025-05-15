from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, Literal

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]
