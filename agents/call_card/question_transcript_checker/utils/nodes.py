from langchain_openai import ChatOpenAI

# Define the function that calls the model
# This is the agent responsible for calling the model
system_prompt = """You are an assistant that analyzes a sales‑call transcript for the sales rep. Identify and list "topics / questions" from a sales call transcript of a customer call and determine which of these "topics / questions" have been asked. 

For each question object, determine:
- question_asked: "Yes" or "No"
- topic_discussed: "Yes" or "No"
- answer_reference: exact timestamp(s) or transcript line(s), or empty string
          
**Output**: a pure JSON array of objects, each with:
- "question_id" (string): matches the input question's id  
- "question" (string): the question text  
- "question_asked" (string): "Yes" or "No"  
- "topic_discussed" (string): "Yes" or "No"  
- "answer_reference" (string): timestamp(s) or empty

Do not output anything else.
"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", seed=42)
    response = model.invoke(messages)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

