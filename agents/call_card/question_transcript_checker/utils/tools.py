from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts.prompt import PromptTemplate
import json
import os
from openai import OpenAI
# from pinecone import Pinecone

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "modulir-agent-development"  # Update this as needed

# Initialize Pinecone
# try:
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     index = pc.Index(INDEX_NAME)
# except Exception as e:
#     print(f"Error initializing Pinecone: {str(e)}")
#     raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

# Add embedding function
# def get_embedding(text, embed_model="text-embedding-3-large"):
#     response = client.embeddings.create(input=[text], model=embed_model)
#     embedding = response.data[0].embedding
#     return embedding

# Load prompts from files
prompt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
critique_prompt = PromptTemplate.from_file(os.path.join(prompt_dir, "critique_prompt.txt"))
estimation_prompt = PromptTemplate.from_file(os.path.join(prompt_dir, "estimation_prompt.txt"))

# Initialize models
internal_model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0
)

anthropic_model = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=1,
    max_tokens=5000,
    thinking={"type": "enabled", "budget_tokens": 2000}
)

@tool
def query_similar_tasks(
    query: str, 
    assignee: str, 
    filter_by_assignee: bool,
    similarity_threshold: float = 0.45,
    enterprise_id: str = "" 
):
    """
    Queries Pinecone to return similar tasks based on the given query.
    This feature is temporarily disabled.
    
    Parameters:
        query (str): The search query "task title: ... ; task description: ... ;".
        assignee (str): The assignee name to filter tasks.
        filter_by_assignee (bool): Flag to filter by assignee.
        similarity_threshold (float): Minimum similarity score (must be >= 0.45).
        enterprise_id (str): Enterprise ID to use as namespace.
    Returns:
        str: A formatted string of similar tasks data.
    """
    return "Similar Tasks:\nNo historic similar tasks found (feature temporarily disabled).\n"
    
    # if enterprise_id == "" or enterprise_id == None:
    #     return "Similar Tasks:\nNo historic similar tasks found.\n"
        
    
    # # Enforce minimum threshold
    # if similarity_threshold < 0.45:
    #     similarity_threshold = 0.45
    
    # # Convert query to embedding
    # query_vector = get_embedding(query)
    
    # # Define metadata filter if needed
    # filter_dict = {"assignee": {"$eq": assignee}} if filter_by_assignee else None
    
    # # Query Pinecone using enterprise_id as namespace
    # response = index.query(
    #     vector=query_vector,
    #     top_k=3,
    #     filter=filter_dict,
    #     namespace=enterprise_id,
    #     include_metadata=True
    # )
    
    # # Count tasks below threshold
    # below_threshold = [match for match in response['matches'] if float(match['score']) < similarity_threshold]
    # total_tasks = len(response['matches'])
    
    # # Format results
    # formatted_tasks = "Similar Tasks:\n"
    
    # # Check if all tasks are below the threshold
    # if len(below_threshold) == total_tasks:
    #     formatted_tasks += "No historic similar tasks found.\n"
    
    # # continue if the below threshold is not empty
    # for match in response['matches']:
    #     metadata = match['metadata']
    #     similarity_score = float(match['score'])
    #     if float(match['score']) < similarity_threshold:
    #         continue
        
    #     formatted_tasks += (
    #         f"- Title: {metadata.get('task-title', 'N/A')}\n"
    #         f"  Story Point: {metadata.get('story-point', 'N/A')}\n"
    #         f"  Assignee: {metadata.get('assignee', 'N/A')}\n"
    #         f"  Description: {metadata.get('description', 'N/A')}\n"
    #         f"  Epic Name: {metadata.get('epic-name', 'N/A')}\n"
    #         f"  Task-type: {metadata.get('issue-type', 'N/A')}\n"
    #         f"  Similarity Score: {similarity_score:.3f}"
    #         f"{' (Below threshold)' if similarity_score < similarity_threshold else ''}\n"
    #     )
    
    # return formatted_tasks

@tool
def estimate_effort(
    title: str,
    description: str,
    developer_profile: Optional[str] = None,
    critique_estimate: str = "No critique provided.",
    historic_data: Optional[str] = None,
    already_completed_tasks_summary: Optional[str] = None,
    epic_name: Optional[str] = "N/A",
    epic_description: Optional[str] = "N/A",
) -> str:
    """
    Estimates task effort using mental models (Probabilistic Thinking, Regression to Mean, Margin of Safety, Circle of Competence)
    
    Parameters:
        title (str): The title of the task to estimate.
        description (str): Detailed description of the task.
        developer_profile (str): Developer's structured skills, experience, and historical performance.
        critique_estimate (str): Critique of the current estimate in form of questions to be added to the estimation prompt.
        historic_data (str): Historic data about the task.
        already_completed_tasks_summary (str): Summary of already completed tasks.
        epic_name (str): Epic name for context.
        epic_description (str): Epic description for context.
    Returns:
        str: A formatted string of the estimated effort.
    """
    
    if developer_profile is None:
        developer_profile = "experience: unknown; skills: []; historical_performance: unknown"
    
    messages = [
        SystemMessage(content="You are an expert story point estimator that uses mental models to generate accurate task estimates. Respond directly with your estimate, do not use any tools."),
        HumanMessage(content="Critique of the current estimate in form of questions to be added to the estimation prompt:\n" + critique_estimate),
        HumanMessage(content=estimation_prompt.format(
            title=title,
            description=description,
            epic_name=epic_name,
            epic_description=epic_description,
            developer_profile=developer_profile,
            historic_data=historic_data,
            already_completed_tasks_summary=already_completed_tasks_summary
        ))
    ]
    
    # Use internal_model that doesn't have tools bound to it
    response = internal_model.invoke(messages)
    return response.content

@tool
def critique_estimate(
    title: str,
    description: str,
    current_estimate: str,
    estimate_effort_rationale: str = None,
    developer_profile: Optional[str] = None,
    already_completed_tasks_summary: Optional[str] = None,
    epic_name: Optional[str] = "N/A",
    epic_description: Optional[str] = "N/A",
) -> str:
    """
    Critique the current effort estimate using historical conversation context.
    Parameters:
        title (str): The title of the task to estimate.
        description (str): Detailed description of the task.
        current_estimate (str): The current estimate of the task.
        estimate_effort_rationale (str): The rationale for the current estimate.
        developer_profile (str): Developer's structured skills, experience, and historical performance.
        already_completed_tasks_summary (str): Summary of already completed tasks.
        epic_name (str): Epic name for context.
        epic_description (str): Epic description for context.
    Returns:
        str: A formatted string of the critique.
    """
    if developer_profile is None:
        developer_profile = "experience: unknown; skills: []; historical_performance: unknown"
    if estimate_effort_rationale is None:
        estimate_effort_rationale = "No prior estimate_effort_rationale provided."
    
    # Construct a messages list that first informs the model about prior chat context.
    messages = [
        SystemMessage(content="You are a seasoned senior developer. Critically evaluate task estimates using all available context, including prior conversation history."),
        HumanMessage(content="Estimation Effort rationale:\n" + estimate_effort_rationale),
        HumanMessage(content=critique_prompt.format(
            title=title,
            description=description,
            developer_profile=developer_profile,
            current_estimate=current_estimate,
            already_completed_tasks_summary=already_completed_tasks_summary
        ))
    ]
    
    response = anthropic_model.invoke(messages)
    return response.content

# Define tools list
tools = [query_similar_tasks, estimate_effort, critique_estimate]