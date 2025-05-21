from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import json

# Define the JSON schema for structured output
class QuestionFormat(TypedDict):
    text: str

class SectionFormat(TypedDict):
    title: str
    durationMinutes: int
    questions: List[QuestionFormat]

class CallCardTemplateResponseFormat(TypedDict):
    name: str
    description: str
    totalDurationMinutes: int
    sections: List[SectionFormat]

# Define the function that calls the model to structure data
def structure_data_node(state, config):
    """
    Calls an LLM to convert raw string data from the last message into structured JSON.
    """
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        messages = state["messages"]
        last_message_content = messages[-1].content
        
        # Bind the schema to the model
        model_with_structure = model.with_structured_output(CallCardTemplateResponseFormat)
        # Invoke the model
        structured_data = model_with_structure.invoke(last_message_content)
    
        return {"messages": [{"role": "assistant", "content": json.dumps(structured_data, indent=2)}]}
    except json.JSONDecodeError:
        # Handle invalid JSON response
        return {"messages": [{"role": "assistant", "content": "Error: Could not parse response as JSON"}]}
    except TypeError as e:
        # Handle schema validation errors
        return {"messages": [{"role": "assistant", "content": f"Error: Response does not match expected schema - {str(e)}"}]} 