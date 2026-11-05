from types import SimpleNamespace

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
    """Test knowledge-base retrieval and source collection."""

    tool_call = create_tool_call(
        "call_1",
        "search_knowledge_base",
        '{"query": "password reset", "top_k": 5}',
    )

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

    executed_tools = []

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append((tool_name, arguments))
        return expected_tool_result

    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        workflow,
        "execute_tool",
        mock_execute_tool,
    )

    result = workflow.run_agent_workflow("How do I reset my password?")

    assert result["answer"] == "Follow the password reset instructions."
    assert result["sources"] == [{"source": "help.pdf"}]
    assert result["total_results"] == 1

    assert executed_tools[0][0] == "search_knowledge_base"
    assert call_count == 2


# ==========================================================
# 3. TEST MULTIPLE TOOL CALLS
# ==========================================================

def test_workflow_executes_multiple_tools(monkeypatch):
    """Test that multiple requested tools are executed."""

    category_call = create_tool_call(
        "call_1",
        "predict_category",
        '{"text": "Unable to access my account"}',
    )

    urgency_call = create_tool_call(
        "call_2",
        "predict_urgency",
        '{"text": "Payment transaction failed"}',
    )

    responses = [
        create_response(
            FakeAssistantMessage(
                tool_calls=[category_call, urgency_call]
            )
        ),
        create_response(
            FakeAssistantMessage(
                content="The category and urgency predictions are ready."
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

    executed_tools = []

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)

        if tool_name == "predict_category":
            return {"category": "account_access"}

        if tool_name == "predict_urgency":
            return {"urgency": "high"}

        raise ValueError("Unexpected tool")

    monkeypatch.setattr(
        workflow,
        "get_openai_client",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        workflow,
        "execute_tool",
        mock_execute_tool,
    )

    result = workflow.run_agent_workflow(
        "Classify this account issue and predict its urgency."
    )

    assert result["answer"] == (
        "The category and urgency predictions are ready."
    )
    assert executed_tools == [
        "predict_category",
        "predict_urgency",
    ]
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