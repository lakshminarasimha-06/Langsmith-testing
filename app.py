import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "agentic-rag-demo")

app = FastAPI(title="Agentic RAG Demo")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- Tools ----
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for company information."""
    kb = {
        "harrison": "Harrison worked at Kensho as a researcher",
        "kensho": "Kensho is a financial analytics company",
        "projects": "Current projects include NLP research and ML model development",
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
    return f"The weather in {location} is sunny with a temperature of 72°F"

# Optional tools
tools = [search_knowledge_base, calculate, get_current_weather]

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    tools.append(
        Tool(
            name="WebSearch",
            func=search.run,
            description="Searches the web for current information using DuckDuckGo",
        )
    )
except Exception as e:
    print(f"Warning: DuckDuckGo search not available: {e}")

try:
    from langchain_community.utilities import WikipediaAPIWrapper
    wikipedia = WikipediaAPIWrapper()
    tools.append(
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Searches Wikipedia for factual information about topics, people, places, etc.",
        )
    )
except Exception as e:
    print(f"Warning: Wikipedia not available: {e}")

try:
    agent_executor = create_react_agent(llm, tools)
except Exception as e:
    agent_executor = None
    print(f"ERROR: Failed to build agent: {e}")

class Query(BaseModel):
    question: str

@traceable(
    name="Agentic RAG Session",
    tags=["agent", "rag", "langgraph"],
    metadata={"model": "gpt-4o-mini"}
)
def run_agent(question: str):
    result = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content

@app.post("/ask")
def ask_agent(query: Query):
    """Ask the LangGraph agent a question."""
    if agent_executor is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Agent failed to initialize. Check container logs.")
    answer = run_agent(query.question)
    return {"question": query.question, "answer": answer}

@app.get("/")
def root():
    return {"message": "Agentic RAG Demo API is running. Use POST /ask with {'question': '...'}"}

@app.get("/health")
def health():
    return {"status": "ok", "agent_ready": agent_executor is not None}
