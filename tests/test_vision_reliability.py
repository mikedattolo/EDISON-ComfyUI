import base64
import io

from PIL import Image


def _tiny_png_data_uri():
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_prepare_vision_image_rejects_fake_base64():
    from services.edison_core.vision_reliability import prepare_vision_image

    info = prepare_vision_image("data:image/png;base64,ZmFrZQ==")

    assert info.ok is False
    assert "not a readable image" in info.error


def test_prepare_vision_image_returns_valid_data_uri():
    from services.edison_core.vision_reliability import prepare_vision_image

    info = prepare_vision_image(_tiny_png_data_uri())

    assert info.ok is True
    assert info.data_uri.startswith("data:image/jpeg;base64,")
    assert info.width == 2
    assert info.height == 2
    assert info.sha256_12


def test_openai_vision_message_builder_rejects_no_images():
    import pytest
    from fastapi import HTTPException
    from services.edison_core.app import OpenAIMessage, _build_openai_vision_messages

    with pytest.raises(HTTPException):
        _build_openai_vision_messages(
            [OpenAIMessage(role="user", content=[{"type": "text", "text": "hello"}])],
            trace_id="vis_test",
        )


def test_openai_vision_message_builder_adds_grounding_prompt():
    from services.edison_core.app import OpenAIMessage, _build_openai_vision_messages

    messages, trace = _build_openai_vision_messages(
        [
            OpenAIMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
                    {"type": "text", "text": "describe the image"},
                ],
            )
        ],
        trace_id="vis_test",
    )

    assert trace["image_count"] == 1
    assert messages[0]["role"] == "system"
    assert "visible evidence" in messages[0]["content"]
    assert messages[1]["content"][0]["type"] == "image_url"


def test_vision_confidence_flags_generic_responses():
    from services.edison_core.vision_reliability import assess_vision_response_confidence

    result = assess_vision_response_confidence("As an AI, I cannot see the image.", image_count=1)

    assert result["confidence"] == "low"
    assert "generic_or_nonvisual_response" in result["flags"]
