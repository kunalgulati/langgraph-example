from langchain_openai import ChatOpenAI

# Define the function that calls the model
# This is the agent responsible for calling the model
system_prompt = """You are an assistant that analyzes a sales‑call transcript.
Identify and list "topics / questions" from a sales call transcript of a customer call and determine which of these "topics / questions" have not been answered. 
          
Output: a pure JSON array of objects, each with questions that have not been answered:
    - "id" (string): matches the input question's id  
    - "question" (string): the question text  
    - "rationale" (string): the rationale why it not been answered, in the current transcript
    
Do not output anything else.
"""

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    response = model.invoke(messages)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

