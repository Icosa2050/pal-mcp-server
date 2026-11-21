"""Integration test for conversation continuation persistence."""

from tools.chat import ChatRequest, ChatTool
from utils.conversation_memory import get_thread
from utils.storage_backend import get_storage_backend


def test_first_response_persisted_in_conversation_history(tmp_path):
    """Ensure the assistant's initial reply is stored for newly created threads."""

    # Clear in-memory storage to avoid cross-test contamination
    storage = get_storage_backend()
    storage._store.clear()  # type: ignore[attr-defined]

    tool = ChatTool()
    request = ChatRequest(
        prompt="First question?",
        model="local-llama",
        working_directory_absolute_path=str(tmp_path),
    )
    response_text = "Here is the initial answer."

    # Mimic the first tool invocation (no continuation_id supplied)
    continuation_data = tool._create_continuation_offer(request, model_info={"model_name": "local-llama"})
    tool._create_continuation_offer_response(
        response_text,
        continuation_data,
        request,
        {"model_name": "local-llama", "provider": "custom"},
    )

    thread_id = continuation_data["continuation_id"]
    thread = get_thread(thread_id)

    assert thread is not None
    assert [turn.role for turn in thread.turns] == ["user", "assistant"]
    assert thread.turns[-1].content == response_text

    # Cleanup storage for subsequent tests
    storage._store.clear()  # type: ignore[attr-defined]


async def _chat(tmp_path, prompt, continuation_id=None):
    import json

    from server import handle_call_tool

    arguments = {
        "prompt": prompt,
        "model": "gemini-2.5-flash",
        "working_directory_absolute_path": str(tmp_path),
    }
    if continuation_id:
        arguments["continuation_id"] = continuation_id
    result = await handle_call_tool("chat", arguments)
    return json.loads(result[0].text)["continuation_offer"]


def test_chat_continuation_sends_and_stores_each_turn_once(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from providers.gemini import GeminiModelProvider

    prompts = []
    questions = ["continuation-marker-alpha", "continuation-marker-beta", "continuation-marker-gamma"]

    def generate_content(self, *, prompt, **kwargs):
        prompts.append(prompt)
        return SimpleNamespace(content=f"assistant-answer-{len(prompts)}", metadata={}, usage={})

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiModelProvider, "generate_content", generate_content)

    async def exercise():
        first = await _chat(tmp_path, questions[0])
        thread_id = first["continuation_id"]
        second = await _chat(tmp_path, questions[1], thread_id)
        third = await _chat(tmp_path, questions[2], thread_id)

        assert [[prompt.count(question) for question in questions] for prompt in prompts] == [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]
        assert prompts[1].count("assistant-answer-1") == 1
        assert prompts[2].count("assistant-answer-1") == 1
        assert prompts[2].count("assistant-answer-2") == 1
        thread = get_thread(thread_id)
        assert [turn.role for turn in thread.turns] == ["user", "assistant"] * 3
        assert [turn.content for turn in thread.turns if turn.role == "user"] == questions
        assert first["remaining_turns"] - second["remaining_turns"] == 2
        assert second["remaining_turns"] - third["remaining_turns"] == 2

    asyncio.run(exercise())
