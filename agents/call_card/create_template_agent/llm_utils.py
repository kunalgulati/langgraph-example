from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Define the function that calls the model to structure data
def structure_data_node(state, config):
    """
    Calls an LLM to convert raw string data from the last message into structured JSON.
    """
    print(f"Structuring data with config: {config}")
    messages = state["messages"]
    last_message_content = messages[-1].content

    # Prepare a specific prompt for data structuring
    structuring_prompt = f"""Please convert the following text into a structured JSON object.
Identify the key pieces of information and represent them in a clear, hierarchical JSON format.
If the text contains a list of items, represent them as a JSON array.
If there are specific entities or attributes, make them keys in the JSON object.
Ensure the output is only the JSON object itself, with no other explanatory text.

Raw text:
{last_message_content}

Structured JSON:
"""
    
    # Get the model
    model_name = config.get('configurable', {}).get("model_name", "openai")
    # We don't want the model to use tools for structuring, so we get a model without tools bound
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Invoke the model with the structuring prompt
    # We expect the AI to respond with the JSON structure directly in its content.
    # We create a new list of messages for this specific call, so it doesn't get tool_calls
    # from previous turns.
    response = model.invoke([{"role": "user", "content": structuring_prompt}])

    # Update the last message with the structured content.
    # Assuming the LLM returns the JSON string directly in response.content
    # We might need to parse it or handle potential errors if the response is not valid JSON.
    # For now, we'll assume it's a valid JSON string.
    updated_messages = messages[:-1] + [response] # Replace last message with structured one

    return {"messages": updated_messages} 