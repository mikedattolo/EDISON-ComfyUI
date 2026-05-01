import ast
from pathlib import Path


def _load_rhino_script_builders():
    source_path = Path(__file__).resolve().parents[1] / "services" / "edison_core" / "business_actions.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    selected_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_build_rhino_script", "_build_rhino_shape_body"}
    ]
    builder_module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(builder_module, str(source_path), "exec"), namespace)
    return namespace["_build_rhino_script"]


def test_rhino_prompt_script_contains_visual_feedback_and_result_manifest():
    build_rhino_script = _load_rhino_script_builders()

    script = build_rhino_script("vase", "demo-vase.3dm")

    assert "result_file" in script
    assert ".result.json" in script
    assert "with open(result_file, 'w') as handle:" in script
    assert "encoding='utf-8'" not in script
    assert "_show_progress('EDISON: drew vase profile'" in script
    assert "_focus_model()" in script
    assert "rs.ZoomSelected()" in script
    assert "_write_result(True, 'Rhino model created successfully.'" in script
    assert "traceback.format_exc()" in script