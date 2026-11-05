
"""Main SupportIQ agent."""

from app.agent.workflow import run_agent_workflow


def run_agent(question: str) -> dict:
    """Process a customer question using the AI agent."""

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Execute the agent's tool-selection and response workflow.
    result = run_agent_workflow(question.strip())

    # Preserve the existing API response structure.
    return {
        "query": question.strip(),
        "answer": result["answer"],
        "sources": result["sources"],
        "total_results": result["total_results"],
    }

