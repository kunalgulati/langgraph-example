from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from agents.modulir.estimate_effort.utils.nodes import call_model, should_continue, tool_node, normalize_story_points
from agents.modulir.estimate_effort.utils.state import AgentState, GraphConfig
from agents.modulir.estimate_effort.utils.agent_utils import structure_data_node
   
# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai", "anthropic"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("normalize", normalize_story_points)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If tools need to be called, we go to the action node
        "continue": "action",
        # If no more tool calls, we go to normalize
        "end": "normalize",
    },
)

# We now add a normal edge from `action` to `agent`.
# This means that after `action` is called, `agent` node is called next.
workflow.add_edge("action", "agent")
# After normalize is done, we end the workflow
workflow.add_edge("normalize", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
