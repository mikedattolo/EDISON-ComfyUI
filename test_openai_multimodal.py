#!/usr/bin/env python3
"""Unit tests for OpenAI multimodal message parsing and routing."""

import asyncio
import sys

from pydantic import ValidationError

sys.path.insert(0, ".")


class _FakeRawRequest:
    async def is_disconnected(self):
        return False


class _FakeVisionModel:
    def __init__(self):
        self.last_messages = None

    def create_chat_completion(self, messages, max_tokens=256, temperature=0.2, top_p=0.9, stream=False):
        self.last_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": "This image looks like a test payload."
                    }
                }
            ]
        }


def test_openai_message_accepts_multimodal_user_content():
    from services.edison_core.app import OpenAIMessage

    msg = OpenAIMessage(
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
            {"type": "text", "text": "describe it"},
        ],
    )
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "image_url"


def test_openai_request_rejects_multimodal_assistant_role():
    from services.edison_core.app import OpenAIChatCompletionRequest

    try:
        OpenAIChatCompletionRequest(
            model="fast",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I should not be multimodal here"},
                    ],
                }
            ],
        )
    except ValidationError:
        return
    raise AssertionError("Expected ValidationError for multimodal assistant message")


def test_openai_chat_completion_preserves_multimodal_blocks_for_vision():
    import services.edison_core.app as app_mod
    from services.edison_core.app import OpenAIChatCompletionRequest, openai_chat_completions

    fake_vision = _FakeVisionModel()

    old_vision = app_mod.llm_vision
    old_vision_code = app_mod.llm_vision_code

    app_mod.llm_vision = fake_vision
    app_mod.llm_vision_code = None

    try:
        req = OpenAIChatCompletionRequest(
            model="fast",
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                        {"type": "text", "text": "describe it"},
                    ],
                }
            ],
        )

        resp = asyncio.run(openai_chat_completions(_FakeRawRequest(), req))

        assert fake_vision.last_messages is not None
        assert isinstance(fake_vision.last_messages[0]["content"], list)
        assert fake_vision.last_messages[0]["content"][0]["type"] == "image_url"
        assert resp.choices[0].message["content"]
        assert resp.model.endswith("vision")
    finally:
        app_mod.llm_vision = old_vision
        app_mod.llm_vision_code = old_vision_code


def test_openai_chat_completion_stream_routes_to_vision_stream():
    import services.edison_core.app as app_mod
    from services.edison_core.app import OpenAIChatCompletionRequest, openai_chat_completions

    fake_vision = _FakeVisionModel()
    captured = {}

    old_vision = app_mod.llm_vision
    old_vision_code = app_mod.llm_vision_code
    old_stream_handler = app_mod.openai_stream_completions

    async def _fake_stream_handler(raw_request, llm, model_name, full_prompt, request, chat_messages=None, is_vision_chat=False):
        captured["llm"] = llm
        captured["model_name"] = model_name
        captured["full_prompt"] = full_prompt
        captured["chat_messages"] = chat_messages
        captured["is_vision_chat"] = is_vision_chat
        return {"ok": True, "stream": True}

    app_mod.llm_vision = fake_vision
    app_mod.llm_vision_code = None
    app_mod.openai_stream_completions = _fake_stream_handler

    try:
        req = OpenAIChatCompletionRequest(
            model="fast",
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                        {"type": "text", "text": "describe it"},
                    ],
                }
            ],
        )

        resp = asyncio.run(openai_chat_completions(_FakeRawRequest(), req))

        assert resp["ok"] is True
        assert captured["llm"] is fake_vision
        assert captured["model_name"] == "vision"
        assert captured["full_prompt"] is None
        assert captured["is_vision_chat"] is True
        assert isinstance(captured["chat_messages"], list)
        assert isinstance(captured["chat_messages"][0]["content"], list)
        assert captured["chat_messages"][0]["content"][0]["type"] == "image_url"
    finally:
        app_mod.llm_vision = old_vision
        app_mod.llm_vision_code = old_vision_code
        app_mod.openai_stream_completions = old_stream_handler


def test_vision_chat_handler_selection():
    """Verify _create_vision_chat_handler picks the correct handler class."""
    from services.edison_core.app import _create_vision_chat_handler
    from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler

    # We can't actually instantiate handlers without a real CLIP file,
    # but we can verify the function exists and the handler classes resolve.
    # Also verify the detection logic by checking import paths.
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        has_qwen = True
    except ImportError:
        has_qwen = False

    # Verify the function is callable
    assert callable(_create_vision_chat_handler)

    # Verify llama-cpp-python 0.3.x no longer accepts clip_model_path in Llama()
    import inspect
    from llama_cpp import Llama
    sig = inspect.signature(Llama.__init__)
    assert "clip_model_path" not in sig.parameters, \
        "clip_model_path should NOT be in Llama.__init__ for 0.3.x — chat_handler is required"
    assert "chat_handler" in sig.parameters, \
        "chat_handler should be in Llama.__init__ params"


if __name__ == "__main__":
    test_openai_message_accepts_multimodal_user_content()
    test_openai_request_rejects_multimodal_assistant_role()
    test_openai_chat_completion_preserves_multimodal_blocks_for_vision()
    test_openai_chat_completion_stream_routes_to_vision_stream()
    test_vision_chat_handler_selection()
    print("✓ test_openai_multimodal passed")
