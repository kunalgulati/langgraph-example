from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from agents.modulir.estimate_effort.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import json

# Add this constant at the top of the file with other imports
MAX_TOOL_ITERATIONS = 2

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """
    for this work, you have to set the model_name in the config
    """
    print(f"Getting model: {model_name}")
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]    
    last_message = messages[-1]
    
    # Check for tool calls in both direct tool_calls attribute and additional_kwargs
    has_tool_calls = (
        getattr(last_message, "tool_calls", None) or 
        last_message.additional_kwargs.get("tool_calls")
    )
    
    # Check if we have a final estimate in the message
    has_final_estimate = (
        "Final Estimate:" in last_message.content and 
        "Rationale for the estimate:" in last_message.content and
        "Rationale for Tool Agreement:" in last_message.content
    )
    
    # If we have a final estimate, we can end
    if has_final_estimate:
        return "end"
    # If there are tool calls, we continue
    elif has_tool_calls:
        return "continue"
    # Otherwise, we continue to get a final estimate
    else:
        return "continue"


# Define the function that calls the model
# This is the agent responsible for calling the model
def call_model(state, config):
    messages = state["messages"]
    system_prompt = SystemMessage(f"""You are an Estimator Agent that performs the following steps:
1. Extract Task Details: Parse the incoming request to extract the task title, assignee name, task description, task type, already completed tasks summary, and any additional details.
2. Query Historical Data: Use the query_similar_tasks tool with the extracted title and assignee (filter_by_assignee=True) to retrieve similar tasks (dummy Pinecone data).
3. Initial Estimation: Call the estimate_effort tool to generate an initial story point estimate, passing along the task details, developer profile, and historic data.
4. Critique the Estimate: Invoke the critique_estimate tool to identify any missing considerations or necessary adjustments.
5. Iterative Refinement:
    - Iteratively alternate calls between estimate_effort and critique_estimate until either the MAX_TOOL_ITERATIONS limit is reached for one or both tools, or until both tools agree on an estimate.
    - If the two tools have differing opinions, continue iterating and addressing the concerns raised by each until consensus is achieved or the iteration limit is met.
6. Final Estimate: Once the critique is complete and the estimate is refined, provide the final estimate and rationale for the estimate for the user in the following format:
    - Final Estimate: [X] story points.
    - Rationale for the estimate: [rationale]
    - Rationale for Tool Agreement: [rationale to stop tool calls]

IMPORTANT: Each tool has a maximum limit of {MAX_TOOL_ITERATIONS} calls. After reaching this limit for a specific tool, you must proceed with the best available results.
- query_similar_tasks: max {MAX_TOOL_ITERATIONS} calls
- estimate_effort: max {MAX_TOOL_ITERATIONS} calls
- critique_estimate: max {MAX_TOOL_ITERATIONS} calls

If you receive a message indicating maximum iterations reached for a tool, proceed with your current best estimate or available information.
""")
    updated_messages = [system_prompt] + messages
    model_name = config.get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(updated_messages)
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)

# Define a new node to normalize story points using LLM
def normalize_story_points(state):
    """
    Uses an LLM to normalize the agent's story point estimate to one of the team's accepted values: [2, 5, 10, 15, 20]
    This node runs after the agent completes its estimation process.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Create a prompt for the LLM to normalize the story points
    system_prompt = SystemMessage(
        content="""You are a Story Point Normalizer. Your task is to:
1. Review the agent's final estimate and rationale
2. Convert the estimate to the nearest appropriate value from the team's accepted story points: [2, 5, 10, 15, 20]
3. Provide a brief explanation for why you selected this normalized value

Be thoughtful about the normalization - consider the complexity described in the rationale, not just the numeric proximity.
"""
    )
    
    user_prompt = HumanMessage(
        content=f"""
The agent has provided the following estimate and rationale:

{last_message.content}

Based on this information, normalize the estimate to one of the team's accepted story point values: [2, 5, 10, 15, 20].

Respond in this format:
- Original Estimate: [extract the numeric value]
- Normalized Estimate: [one of: 2, 5, 10, 15, 20]
- Rationale for normalization: [brief explanation]
"""
    )
    
    # Define the JSON schema for the structured response
    json_schema = {
        "title": "NormalizedStoryPoint",
        "description": "Normalized story point estimation",
        "type": "object",
        "properties": {
            "original_estimate": {
                "type": "integer",
                "description": "The original story point estimate provided by the agent"
            },
            "normalized_estimate": {
                "type": "integer",
                "description": "The normalized story point estimate"
            },
            "rationale": {
                "type": "string",
                "description": "Explanation for the normalization decision"
            }
        },
        "required": ["original_estimate", "normalized_estimate", "rationale"]
    }

    # Use the gpt_4o_model with structured output
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    structured_llm = model.with_structured_output(json_schema)
    
    # Get the structured response
    structured_response = structured_llm.invoke([system_prompt, user_prompt])
    
    # Return the structured response directly
    return {"messages": [AIMessage(content=json.dumps(structured_response))]}