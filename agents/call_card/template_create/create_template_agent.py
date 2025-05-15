from typing import TypedDict, Literal, Any
import json

from langgraph.graph import StateGraph, END
from utils.state import AgentState

# --- Configuration schema ---
class GraphConfig(TypedDict):
    model_name: Literal["openai", "anthropic"]

# --- Node implementations ---
def generate_raw_template(state: Any, config: GraphConfig) -> Any:
    """
    First LLM call: generate an unstructured/raw sales call template from input/context.
    """
    prompt = (
        "Create a draft sales call template based on the following input and context, without any specific structure."
        f"\nInput: {state['user_input']}\nContext: {state['context']}\n"
    )
    raw = state['llm'].generate(
        prompt=prompt,
        model_name=config["model_name"]
    )
    state['raw_template'] = raw
    return state


def normalize_structure(state: Any, config: GraphConfig) -> Any:
    """
    Second LLM call: take the raw template and produce a normalized JSON object matching the APIResponse schema.
    """
    prompt = (
        "Normalize the following raw sales call draft into a JSON object with the schema:"
        " { name, description, totalDurationMinutes, sections: [{ title, durationMinutes, questions:[{text}] }] }"
        f"\n\nRaw template:\n{state['raw_template']}"
    )
    normalized = state['llm'].generate(
        prompt=prompt,
        model_name=config["model_name"]
    )
    try:
        structured = json.loads(normalized)
    except json.JSONDecodeError:
        structured = {
            "name": f"AI Template: {state['context']}",
            "description": state['raw_template'],
            "totalDurationMinutes": 0,
            "sections": []
        }
    state['output'] = structured
    return state

# --- Graph definition ---
graph_def = StateGraph(AgentState, config_schema=GraphConfig)

graph_def.add_node("generate_raw", generate_raw_template)
graph_def.add_node("normalize", normalize_structure)

graph_def.set_entry_point("generate_raw")

graph_def.add_edge("generate_raw", "normalize")
graph_def.add_edge("normalize", END)

# Compile into a runnable graph
graph = graph_def.compile()  # type: ignore
