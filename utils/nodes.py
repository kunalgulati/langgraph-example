from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from utils.tools import tools
from langgraph.prebuilt import ToolNode


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """
    for this work, you have to set the model_name in the config
    """
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

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
    
    # If there are no tool calls, then we finish
    if not has_tool_calls:
        return "end"
    # Otherwise if there are tool calls, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant that generate a call card template based on the user's request. Call the normalize node when finished generate template"""

# Define the function that calls the model
def call_model(state, config):
    
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)