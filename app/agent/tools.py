
"""Tools available to the SupportIQ AI agent."""

# JSON-compatible tool definitions are sent to the OpenAI API.
from app.rag.retriever import retrieve_relevant_context
from ml.predict import predict_category
from ml.predict_urgency import predict_urgency


def search_knowledge_base(query: str, top_k: int = 5) -> dict:
    """Search the knowledge base for information relevant to a question."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if (
    isinstance(top_k, bool)
    or not isinstance(top_k, int)
    or not 1 <= top_k <= 10
):
        raise ValueError("top_k must be between 1 and 10.")

    # Retrieve relevant chunks from the existing ChromaDB pipeline.
    result = retrieve_relevant_context(
        query=query.strip(),
        top_k=top_k,
    )

    return result


def predict_category_tool(text: str) -> dict:
    """Predict the category of a customer support issue."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    return {
        "category": predict_category(text.strip()),
    }


def predict_urgency_tool(text: str) -> dict:
    """Predict the urgency of a customer support issue."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    return {
        "urgency": predict_urgency(text.strip()),
    }


# These definitions tell the LLM which tools are available
# and what arguments each tool requires.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search company documentation and support guides "
                "to answer customer questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The information to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant results.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_category_tool",
            "description": (
                "Predict the category of a customer support issue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Customer issue description.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_urgency_tool",
            "description": (
                "Predict the urgency level of a customer support issue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Customer issue description.",
                    },
                },
                "required": ["text"],
            },
        },
    },
]


# Explicit allowlist prevents the LLM from executing arbitrary functions.
AVAILABLE_TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "predict_category_tool": predict_category_tool,
    "predict_urgency_tool": predict_urgency_tool,
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Validate and execute an approved agent tool."""

    tool_function = AVAILABLE_TOOLS.get(tool_name)

    if tool_function is None:
        raise ValueError(f"Unknown agent tool: {tool_name}")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    # Only explicitly registered functions can be executed.
    return tool_function(**arguments)

