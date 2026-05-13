import json


def test_workflow_template_discovery_and_validation(tmp_path):
    from services.edison_core.comfyui_integration import discover_workflow_templates

    template = {
        "edison": {
            "template_id": "persona-basic",
            "name": "Persona Basic",
            "version": "1.0",
            "required_nodes": ["VideoHelperSuite"],
            "required_models": ["checkpoint.safetensors"],
        },
        "workflow": {
            "1": {
                "class_type": "LoadVideo",
                "inputs": {"path": "${source_segment_path}"},
            },
            "2": {
                "class_type": "SaveVideo",
                "inputs": {"filename_prefix": "${output_path}"},
            },
        },
    }
    (tmp_path / "persona.json").write_text(json.dumps(template), encoding="utf-8")

    templates = discover_workflow_templates(tmp_path)

    assert len(templates) == 1
    assert templates[0].template_id == "persona-basic"
    assert templates[0].validation.ok is True
    assert "source_segment_path" in templates[0].validation.placeholders
    assert templates[0].required_nodes == ["VideoHelperSuite"]


def test_workflow_variable_injection_recurses():
    from services.edison_core.comfyui_integration import find_placeholders, inject_workflow_variables

    workflow = {
        "node": {
            "inputs": {
                "video": "${source_segment_path}",
                "meta": ["${segment_id}", "${gpu_index}"],
            }
        }
    }

    assert find_placeholders(workflow) == {"source_segment_path", "segment_id", "gpu_index"}
    injected = inject_workflow_variables(
        workflow,
        {
            "source_segment_path": "/tmp/source.mp4",
            "segment_id": "seg_001",
            "gpu_index": 0,
        },
    )

    assert injected["node"]["inputs"]["video"] == "/tmp/source.mp4"
    assert injected["node"]["inputs"]["meta"] == ["seg_001", "0"]


def test_invalid_workflow_is_reported_without_comfyui(tmp_path):
    from services.edison_core.comfyui_integration import discover_workflow_templates

    (tmp_path / "bad.json").write_text("{not-json", encoding="utf-8")

    templates = discover_workflow_templates(tmp_path)

    assert templates[0].validation.ok is False
    assert "invalid JSON" in templates[0].validation.errors[0]
