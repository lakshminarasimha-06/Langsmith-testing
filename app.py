import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# Ensure LangSmith environment variables are set
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agentic-rag-demo"  # Optional: organize traces by project

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define custom tools with detailed descriptions
@tool
def search_knowledge_base(query: str) -> str:
    """Searches the internal knowledge base for information about employees and company data."""
    # Simulate a knowledge base
    kb = {
        "harrison": "Harrison worked at Kensho as a researcher",
        "kensho": "Kensho is a financial analytics company",
        "projects": "Current projects include NLP research and ML model development"
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return "No information found in knowledge base"

@tool
def calculate(expression: str) -> str:
    """Performs mathematical calculations. Input should be a valid Python expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def get_current_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    # Mock weather data
    return f"The weather in {location} is sunny with a temperature of 72°F"

# Initialize additional tools
try:
    search = DuckDuckGoSearchRun()
    web_search_tool = Tool(
        name="WebSearch",
        func=search.run,
        description="Searches the web for current information using DuckDuckGo"
    )
except Exception as e:
    print(f"Warning: DuckDuckGo search not available: {e}")
    web_search_tool = None

try:
    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Searches Wikipedia for factual information about topics, people, places, etc."
    )
except Exception as e:
    print(f"Warning: Wikipedia not available: {e}")
    wikipedia_tool = None

# Create tool list
tools = [
    search_knowledge_base,
    calculate,
    get_current_weather,
]

# Add optional tools if available
if web_search_tool:
    tools.append(web_search_tool)
if wikipedia_tool:
    tools.append(wikipedia_tool)

# Create the agent using LangGraph
agent_executor = create_react_agent(llm, tools)

@traceable(
    name="Agentic RAG Session",
    tags=["agent", "rag", "langgraph"],
    metadata={"model": "gpt-4o-mini"}
)
def run_agent(question: str):
    """
    Runs the agent with detailed tracing of each step.
    Each tool call and reasoning step will be traced in LangSmith.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Run the agent
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    
    # Extract the final response
    final_response = result["messages"][-1].content
    
    print(f"\n{'='*60}")
    print(f"Final Answer: {final_response}")
    print(f"{'='*60}\n")
    
    return final_response

# @traceable(name="Batch Agent Queries")
# def run_multiple_queries(questions: list[str]):
#     """Run multiple queries and trace them as a batch operation."""
#     results = []
#     for i, question in enumerate(questions, 1):
#         print(f"\n--- Query {i}/{len(questions)} ---")
#         result = run_agent(question)
#         results.append({"question": question, "answer": result})
#     return results


if __name__ == "__main__":
    # Single query example
    print("=== Single Query Demo ===")
    run_agent("Where did virat kohli play cricket?")
    
    # # Multiple queries example
    # print("\n\n=== Multiple Queries Demo ===")
    # questions = [
    #     "Where did Harrison work?",
    #     "What is 157 * 89?",
    #     "What's the weather like in San Francisco?",
    # ]
    # run_multiple_queries(questions)
