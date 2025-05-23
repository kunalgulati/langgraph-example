from functools import lru_cache
from langchain_openai import ChatOpenAI

# Define the function that calls the model
# This is the agent responsible for calling the model
system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    response = model.invoke(messages)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

