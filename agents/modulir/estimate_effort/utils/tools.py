from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os
from openai import OpenAI
from pinecone import Pinecone

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "modulir-agent-development"  # Update this as needed

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"Error initializing Pinecone: {str(e)}")
    raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

# Add embedding function
def get_embedding(text, embed_model="text-embedding-3-large"):
    response = client.embeddings.create(input=[text], model=embed_model)
    embedding = response.data[0].embedding
    return embedding

# Define prompts
critique_prompt = """You are a seasoned senior developer and a key member of the engineering team. Your role is to critically evaluate task effort estimates by leveraging your deep experience and knowledge of the codebase, project history, and industry best practices.

Task Title: {title}
Task Description: {description}
Epic Context:
    - Epic Name: {epic_name}
    - Epic Description: {epic_description}
Developer Profile: {developer_profile}
Current Estimate: {current_estimate}
Already completed tasks: {already_completed_tasks_summary}

Please review the above information and provide a detailed critique for over and under estimation of the effort required to complete the task.
Be direct, constructive, and precise. Your feedback should help the team understand what gaps exist in the current effort estimation and guide them toward a more informed and realistic estimate.

Final output:
- critique of the current estimate
- Suggested estimate: [X] story points. 

Guidelines:
- choose story points from the following scale of 1 to 20

Critique criteria:
- Do not include considerations related to environment-specific configurations, deployments, permissions, security, documentation, testing strategies, dependency management, logging, or migration processes for non-production environments.
"""

estimation_prompt = """Your role is to role-play as the assigned developer and evaluate task effort estimates by leveraging your deep experience and knowledge of the codebase and project. 
Given the following task description and its associated epic, provide an estimated story point considering past similar tasks that have already been completed. 
Use the epic's context to understand dependencies and overall scope. If the task appears redundant based on completed work, flag it as possibly unnecessary

Task Title: {title}
Task Description: {description}
Epic Context:
    - Epic Name: {epic_name}
    - Epic Description: {epic_description}
Similar historical tasks (semantic matches): {historic_data}
Developer Profile (Please provide structured details like years of experience, familiarity with tech stack (1-5), historical estimation accuracy, etc.): {developer_profile}
Completed Tasks in the Epic: 
{already_completed_tasks_summary}

Please review the above information and provide a detailed thought process for your estimation of the effort required to complete the task by assigned developer.
Be direct, constructive, and precise. Your feedback should help the team understand effort required to complete the task and guide them toward a more informed and realistic estimate.

Considering the factors above, developer skills, and historical data, recommend an appropriate story point estimate:

Recommended estimate: [X] story points. 

Guidelines:
- choose story points from the following scale of 1 to 20
"""

# Initialize models
internal_model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0
)

anthropic_model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
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
    
    Parameters:
        query (str): The search query "task title: ... ; task description: ... ;".
        assignee (str): The assignee name to filter tasks.
        filter_by_assignee (bool): Flag to filter by assignee.
        similarity_threshold (float): Minimum similarity score (must be >= 0.45).
        enterprise_id (str): Enterprise ID to use as namespace.
    Returns:
        str: A formatted string of similar tasks data.
    """
    
    if enterprise_id == "" or enterprise_id == None:
        return "Similar Tasks:\nNo historic similar tasks found.\n"
        
    
    # Enforce minimum threshold
    if similarity_threshold < 0.45:
        similarity_threshold = 0.45
    
    # Convert query to embedding
    query_vector = get_embedding(query)
    
    # Define metadata filter if needed
    filter_dict = {"assignee": {"$eq": assignee}} if filter_by_assignee else None
    
    # Query Pinecone using enterprise_id as namespace
    response = index.query(
        vector=query_vector,
        top_k=3,
        filter=filter_dict,
        namespace=enterprise_id,
        include_metadata=True
    )
    
    # Count tasks below threshold
    below_threshold = [match for match in response['matches'] if float(match['score']) < similarity_threshold]
    total_tasks = len(response['matches'])
    
    # Format results
    formatted_tasks = "Similar Tasks:\n"
    
    # Check if all tasks are below the threshold
    if len(below_threshold) == total_tasks:
        formatted_tasks += "No historic similar tasks found.\n"
    
    # continue if the below threshold is not empty
    for match in response['matches']:
        metadata = match['metadata']
        similarity_score = float(match['score'])
        if float(match['score']) < similarity_threshold:
            continue
        
        formatted_tasks += (
            f"- Title: {metadata.get('task-title', 'N/A')}\n"
            f"  Story Point: {metadata.get('story-point', 'N/A')}\n"
            f"  Assignee: {metadata.get('assignee', 'N/A')}\n"
            f"  Description: {metadata.get('description', 'N/A')}\n"
            f"  Epic Name: {metadata.get('epic-name', 'N/A')}\n"
            f"  Task-type: {metadata.get('issue-type', 'N/A')}\n"
            f"  Similarity Score: {similarity_score:.3f}"
            f"{' (Below threshold)' if similarity_score < similarity_threshold else ''}\n"
        )
    
    return formatted_tasks

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
            epic_name=epic_name,
            epic_description=epic_description,
            developer_profile=developer_profile,
            current_estimate=current_estimate,
            already_completed_tasks_summary=already_completed_tasks_summary
        ))
    ]
    
    response = anthropic_model.invoke(messages)
    return response.content

# Define tools list
tools = [query_similar_tasks, estimate_effort, critique_estimate]