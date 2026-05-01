import importlib.util
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch


def _load_node_agent_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "node-agent" / "edison_node_agent.py"
    spec = importlib.util.spec_from_file_location("edison_node_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    fake_requests = types.ModuleType("requests")
    fake_requests.post = None
    fake_requests.packages = types.SimpleNamespace(
        urllib3=types.SimpleNamespace(disable_warnings=lambda *args, **kwargs: None)
    )
    fake_requests.exceptions = types.SimpleNamespace(RequestException=Exception)

    with patch.dict(sys.modules, {"requests": fake_requests}):
        spec.loader.exec_module(module)
    return module


def test_rhino_controller_uses_thread_local_com_sessions():
    module = _load_node_agent_module()

    class FakeRhinoApp:
        def __init__(self, identifier):
            self.identifier = identifier

        def GetScriptObject(self):
            return object()

    class FakeWin32Client:
        def __init__(self):
            self.apps = []

        def Dispatch(self, _progid):
            app = FakeRhinoApp(len(self.apps) + 1)
            self.apps.append(app)
            return app

    class FakePythonCom:
        def __init__(self):
            self.calls = []

        def CoInitialize(self):
            self.calls.append(threading.get_ident())

    controller = module.RhinoController.__new__(module.RhinoController)
    controller.available = True
    controller._thread_state = threading.local()
    controller._win32_client = FakeWin32Client()
    controller._pythoncom = FakePythonCom()

    main_rhino, _ = controller._get_session()
    same_thread_rhino, _ = controller._get_session()
    other_thread_rhinos = []

    def worker():
        rhino, _ = controller._get_session()
        other_thread_rhinos.append(rhino)

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    worker_thread.join()

    assert main_rhino is same_thread_rhino
    assert other_thread_rhinos[0] is not main_rhino
    assert len(controller._win32_client.apps) == 2
    assert len(controller._pythoncom.calls) == 2


def test_rhino_controller_falls_back_to_script_object_runscript(tmp_path):
    module = _load_node_agent_module()

    class FailingRhinoApp:
        def RunScript(self, _command, _echo):
            raise RuntimeError("Rhino.Application.RunScript")

    class ScriptObject:
        def __init__(self):
            self.calls = []

        def RunScript(self, command, echo):
            self.calls.append((command, echo))

    script_file = tmp_path / "inline.py"
    script_file.write_text("print('hello')", encoding="utf-8")

    script_object = ScriptObject()
    controller = module.RhinoController.__new__(module.RhinoController)
    controller.available = True
    controller._get_session = lambda: (FailingRhinoApp(), script_object)

    result = module.RhinoController.run_python_script(controller, str(script_file))

    assert result["ok"] is True
    assert result["transport"] == "script_object"
    assert result["script"] == str(script_file)
    assert script_object.calls == [(f'-_RunPythonScript "{script_file}"', 0)]


def test_rhino_controller_keeps_inline_script_file_until_rhino_can_read_it(tmp_path, monkeypatch):
    module = _load_node_agent_module()
    monkeypatch.setenv("HOME", str(tmp_path))

    controller = module.RhinoController.__new__(module.RhinoController)
    controller.available = True
    controller._cleanup_stale_inline_scripts = lambda *_args, **_kwargs: None
    controller.run_python_script = lambda script_path: {"ok": True, "script": script_path, "transport": "application"}

    result = module.RhinoController.run_python_code(
        controller,
        "print('hello from rhino')",
        output_paths=["~/.edison/generated_models/vase.3dm"],
    )

    script_path = Path(result["script_path"])
    assert result["ok"] is True
    assert result["output_paths"] == ["~/.edison/generated_models/vase.3dm"]
    assert script_path.exists()
    assert script_path.read_text(encoding="utf-8") == "print('hello from rhino')"