from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from agents.call_card.question_transcript_checker.utils.nodes import call_model
from agents.call_card.question_transcript_checker.utils.state import AgentState, GraphConfig
   
# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("agent", call_model)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a normal edge from `agent` to END.
# This means that after `agent` is called, the workflow is ended.
workflow.add_edge("agent", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
