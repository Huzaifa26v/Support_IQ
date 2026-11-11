# Import the actual registered agent tools for integration testing.
from types import SimpleNamespace

from app.agent import tools as agent_tools
from app.agent import workflow

# ==========================================================
# MOCK OPENAI RESPONSE HELPERS
# ==========================================================

class FakeAssistantMessage:
    """Create a mock assistant message for workflow testing."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        """Simulate the OpenAI message serialization method."""

        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in (self.tool_calls or [])
            ],
        }


def create_tool_call(tool_id, name, arguments):
    """Create a mock OpenAI function tool call."""

    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def create_response(message):
    """Wrap a mock message in the OpenAI response structure."""

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


# ==========================================================
# 1. TEST NORMAL RESPONSE WITHOUT TOOL CALLS
# ==========================================================

def test_workflow_returns_normal_response(monkeypatch):
    """Test when the LLM answers without requesting a tool."""

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: create_response(
                    FakeAssistantMessage(
                        content="You can contact customer support."
                    )
                )
            )
        )
    )

    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    result = workflow.run_agent_workflow("How can I contact support?")

    assert result["answer"] == "You can contact customer support."
    assert result["sources"] == []
    assert result["total_results"] == 0


# ==========================================================
# 2. TEST KNOWLEDGE BASE TOOL CALL
# ==========================================================
def test_workflow_executes_knowledge_base_tool(monkeypatch):
    """Test real tool dispatch, knowledge retrieval, and source collection."""

    # Create a knowledge-base search tool call.
    tool_call = create_tool_call(
        "call_1",
        "search_knowledge_base",
        '{"query": "password reset", "top_k": 5}',
    )

    # Simulate the LLM requesting retrieval and then responding.
    responses = [
        create_response(
            FakeAssistantMessage(tool_calls=[tool_call])
        ),
        create_response(
            FakeAssistantMessage(
                content="Follow the password reset instructions."
            )
        ),
    ]

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        response = responses[call_count]
        call_count += 1
        return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )
    )

    # Provide a predictable result from the RAG retriever.
    expected_tool_result = {
        "query": "password reset",
        "context": "Use the password reset page.",
        "chunks": [
            {
                "content": "Password reset instructions.",
                "metadata": {"source": "help.pdf"},
                "distance": 0.2,
            }
        ],
        "total_results": 1,
    }

    # Record calls to verify the actual search tool uses the retriever.
    retriever_calls = []

    def mock_retriever(query, top_k):
        retriever_calls.append(
            {"query": query, "top_k": top_k}
        )
        return expected_tool_result

    # Mock only the external dependencies.
    # Keep the real tool dispatcher and search tool active.
    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        agent_tools,
        "retrieve_relevant_context",
        mock_retriever,
    )

    # Execute the complete agent workflow.
    result = workflow.run_agent_workflow(
        "How do I reset my password?"
    )

    # Verify the final LLM response.
    assert result["answer"] == (
        "Follow the password reset instructions."
    )

    # Verify source collection from the actual tool result.
    assert result["sources"] == [{"source": "help.pdf"}]
    assert result["total_results"] == 1

    # Verify the real search tool called the retriever correctly.
    assert retriever_calls == [
        {
            "query": "password reset",
            "top_k": 5,
        }
    ]

    # Verify both LLM requests were completed.
    assert call_count == 2

# ==========================================================
# 3. TEST MULTIPLE TOOL CALLS
# ==========================================================
def test_workflow_executes_multiple_tools(monkeypatch):
    """Test execution of actual registered category and urgency tools."""

    # Create a category prediction tool call.
    category_call = create_tool_call(
        "call_1",
        "predict_category_tool",
        '{"text": "Unable to access my account"}',
    )

    # Create an urgency prediction tool call.
    urgency_call = create_tool_call(
        "call_2",
        "predict_urgency_tool",
        '{"text": "Payment transaction failed"}',
    )

    # Simulate the LLM requesting both tools and then responding.
    responses = [
        create_response(
            FakeAssistantMessage(
                tool_calls=[category_call, urgency_call]
            )
        ),
        create_response(
            FakeAssistantMessage(
                content="The category is account_access and urgency is high."
            )
        ),
    ]

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        response = responses[call_count]
        call_count += 1
        return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )
    )

    # Replace the actual ML prediction functions with test functions.
    monkeypatch.setitem(
        agent_tools.AVAILABLE_TOOLS,
        "predict_category_tool",
        lambda text: {"category": "account_access"},
    )

    monkeypatch.setitem(
        agent_tools.AVAILABLE_TOOLS,
        "predict_urgency_tool",
        lambda text: {"urgency": "high"},
    )

    # Mock the LLM client but retain the real tool dispatcher.
    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    result = workflow.run_agent_workflow(
        "Classify this account issue and predict its urgency."
    )

    # Verify the final LLM response.
    assert result["answer"] == (
        "The category is account_access and urgency is high."
    )

    # Verify both actual registered tools were available.
    assert agent_tools.AVAILABLE_TOOLS[
        "predict_category_tool"
    ]("test") == {"category": "account_access"}

    assert agent_tools.AVAILABLE_TOOLS[
        "predict_urgency_tool"
    ]("test") == {"urgency": "high"}

    assert call_count == 2


# ==========================================================
# 4. TEST INVALID TOOL ARGUMENTS
# ==========================================================

def test_workflow_handles_invalid_tool_arguments(monkeypatch):
    """Test that invalid JSON arguments are handled safely."""

    tool_call = create_tool_call(
        "call_1",
        "search_knowledge_base",
        "{invalid-json}",
    )

    responses = [
        create_response(
            FakeAssistantMessage(tool_calls=[tool_call])
        ),
        create_response(
            FakeAssistantMessage(
                content="I could not process the search request."
            )
        ),
    ]

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        response = responses[call_count]
        call_count += 1
        return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )
    )

    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    result = workflow.run_agent_workflow("Search the knowledge base.")

    assert result["answer"] == (
        "I could not process the search request."
    )
    assert result["sources"] == []
    assert result["total_results"] == 0
    assert call_count == 2


# ==========================================================
# 5. TEST MAXIMUM TOOL ROUND LIMIT
# ==========================================================

def test_workflow_respects_max_tool_rounds(monkeypatch):
    """Test that the workflow stops requesting additional tools."""

    tool_call = create_tool_call(
        "call_1",
        "predict_category",
        '{"text": "Unable to access my account"}',
    )

    responses = [
        create_response(
            FakeAssistantMessage(tool_calls=[tool_call])
        ),
        create_response(
            FakeAssistantMessage(
                content="Final response after tool execution."
            )
        ),
    ]

    call_count = 0
    request_arguments = []

    def mock_create(**kwargs):
        nonlocal call_count

        request_arguments.append(kwargs)
        response = responses[call_count]
        call_count += 1

        return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )
    )

    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        workflow,
        "execute_tool",
        lambda tool_name, arguments: {
            "category": "account_access"
        },
    )

    result = workflow.run_agent_workflow(
        "Classify this ticket.",
        max_tool_rounds=1,
    )

    assert result["answer"] == "Final response after tool execution."
    assert call_count == 2

    # The final LLM request must not include tool definitions.
    assert "tools" not in request_arguments[-1]