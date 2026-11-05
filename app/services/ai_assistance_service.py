
# Import the new SupportIQ AI agent.
from app.agent.agent import run_agent


def ask_question(question: str) -> dict:
    """
    Process a customer question using the SupportIQ AI agent.

    The agent decides which tools to use, such as knowledge-base
    search, category prediction, and urgency prediction.

    Args:
        question: Customer's support question.

    Returns:
        Dictionary containing the answer and supporting sources.
    """

    # Validate the customer's question.
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Delegate question processing to the AI agent.
    result = run_agent(question.strip())

    # Return the agent's response.
    return result

