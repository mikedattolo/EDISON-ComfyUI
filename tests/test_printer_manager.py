import json
from pathlib import Path

from services.edison_core.printer import PrinterManager


class _Resp:
    def __init__(self, ok=True, status=200, body=None):
        self.ok = ok
        self.status_code = status
        self._body = body or {"state": "idle"}
        self.headers = {"content-type": "application/json"}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_printer_manager_list_and_status(tmp_path, monkeypatch):
    db = tmp_path / "printers.json"
    db.write_text(json.dumps({"printers": [{"id": "p1", "name": "Bambu", "type": "bambu", "host": "192.168.1.50", "enabled": True}]}))
    pm = PrinterManager(db_path=db, workspace_root=tmp_path)

    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(ok=True, body={"state": "ready"}))

    listed = pm.list_printers()
    assert listed["printers"][0]["id"] == "p1"

    status = pm.get_printer_status("p1")
    assert status["state"] == "ready"


def test_printer_manager_send_print(tmp_path, monkeypatch):
    db = tmp_path / "printers.json"
    db.write_text(json.dumps({"printers": [{"id": "p1", "name": "Bambu", "type": "bambu", "host": "192.168.1.50", "enabled": True}]}))
    gcode = tmp_path / "part.gcode"
    gcode.write_text("G1 X0 Y0")

    pm = PrinterManager(db_path=db, workspace_root=tmp_path)
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(ok=True, body={"job": "queued"}))

    result = pm.send_3d_print("p1", str(gcode))
    assert result["success"] is True
