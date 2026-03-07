"""3D printer integration layer for Edison.

Provides a manager + driver abstraction so API/tools can work with multiple
printer backends while sharing a single config store.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class PrinterProfile:
    id: str
    name: str
    type: str
    host: str = ""
    endpoint: str = ""
    api_key: str = ""
    enabled: bool = True


class PrinterDriver:
    """Base printer driver interface."""

    def send_3d_print(self, profile: PrinterProfile, gcode_path: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def get_printer_status(self, profile: PrinterProfile) -> Dict[str, Any]:
        raise NotImplementedError


class BambuLabDriver(PrinterDriver):
    """Best-effort Bambu LAN driver using Bambu Connect-style HTTP endpoints."""

    def _base_url(self, profile: PrinterProfile) -> str:
        if profile.endpoint:
            return profile.endpoint.rstrip("/")
        host = profile.host.strip()
        if not host:
            raise ValueError("Bambu printer is missing host/endpoint")
        return f"http://{host}"

    def send_3d_print(self, profile: PrinterProfile, gcode_path: Path) -> Dict[str, Any]:
        if not gcode_path.exists():
            raise FileNotFoundError(f"G-code not found: {gcode_path}")

        base = self._base_url(profile)
        headers = {}
        if profile.api_key:
            headers["Authorization"] = f"Bearer {profile.api_key}"

        files = {"file": (gcode_path.name, gcode_path.read_bytes(), "application/octet-stream")}
        payload = {"filename": gcode_path.name, "start": True}

        # Primary endpoint used by Bambu Connect bridges.
        resp = requests.post(
            f"{base}/api/v1/print/jobs",
            headers=headers,
            files=files,
            data=payload,
            timeout=30,
        )

        # Fallback for common custom LAN bridge setups.
        if resp.status_code == 404:
            resp = requests.post(
                f"{base}/upload",
                headers=headers,
                files=files,
                data=payload,
                timeout=30,
            )

        if not resp.ok:
            raise RuntimeError(f"Bambu upload failed ({resp.status_code}): {resp.text[:300]}")

        return {
            "success": True,
            "message": f"Sent {gcode_path.name} to Bambu printer",
            "printer_id": profile.id,
            "response": resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text[:400],
        }

    def get_printer_status(self, profile: PrinterProfile) -> Dict[str, Any]:
        base = self._base_url(profile)
        headers = {}
        if profile.api_key:
            headers["Authorization"] = f"Bearer {profile.api_key}"

        resp = requests.get(f"{base}/api/v1/printer/status", headers=headers, timeout=10)
        if resp.status_code == 404:
            resp = requests.get(f"{base}/status", headers=headers, timeout=10)

        if not resp.ok:
            raise RuntimeError(f"Failed to read printer status ({resp.status_code})")

        data: Dict[str, Any]
        if "application/json" in resp.headers.get("content-type", ""):
            data = resp.json()
        else:
            data = {"raw": resp.text[:400]}

        state = data.get("state") or data.get("status") or "unknown"
        return {
            "success": True,
            "printer_id": profile.id,
            "state": state,
            "details": data,
        }


class OctoPrintDriver(PrinterDriver):
    """OctoPrint-compatible fallback driver."""

    def send_3d_print(self, profile: PrinterProfile, gcode_path: Path) -> Dict[str, Any]:
        endpoint = profile.endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("OctoPrint profile requires endpoint")
        headers = {"X-Api-Key": profile.api_key} if profile.api_key else {}
        files = {"file": (gcode_path.name, gcode_path.read_bytes(), "application/octet-stream")}
        resp = requests.post(f"{endpoint}/api/files/local", headers=headers, files=files, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"OctoPrint upload failed ({resp.status_code})")
        return {"success": True, "message": f"Uploaded {gcode_path.name}", "printer_id": profile.id}

    def get_printer_status(self, profile: PrinterProfile) -> Dict[str, Any]:
        endpoint = profile.endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("OctoPrint profile requires endpoint")
        headers = {"X-Api-Key": profile.api_key} if profile.api_key else {}
        resp = requests.get(f"{endpoint}/api/printer", headers=headers, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"OctoPrint status failed ({resp.status_code})")
        data = resp.json()
        state = data.get("state", {}).get("text", "unknown")
        return {"success": True, "printer_id": profile.id, "state": state, "details": data}


class GenericStubDriver(PrinterDriver):
    """Stub fallback for unsupported printer types."""

    def send_3d_print(self, profile: PrinterProfile, gcode_path: Path) -> Dict[str, Any]:
        return {
            "success": False,
            "printer_id": profile.id,
            "message": f"Printer type '{profile.type}' is not implemented yet",
        }

    def get_printer_status(self, profile: PrinterProfile) -> Dict[str, Any]:
        return {
            "success": True,
            "printer_id": profile.id,
            "state": "unknown",
            "details": {"message": f"No status driver for type '{profile.type}'"},
        }


class PrinterManager:
    def __init__(self, db_path: Path, workspace_root: Optional[Path] = None):
        self._db_path = db_path
        self._workspace_root = workspace_root or Path.cwd()
        self._drivers: Dict[str, PrinterDriver] = {
            "bambu": BambuLabDriver(),
            "octoprint": OctoPrintDriver(),
            "generic": GenericStubDriver(),
            "prusa": GenericStubDriver(),
            "creality": GenericStubDriver(),
            "voron": GenericStubDriver(),
        }
        self._ensure_db()

    def _ensure_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._db_path.exists():
            self._db_path.write_text(json.dumps({"printers": []}, indent=2))

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self._db_path.read_text())
        except Exception:
            return {"printers": []}

    def _save(self, data: Dict[str, Any]):
        self._db_path.write_text(json.dumps(data, indent=2))

    def _to_profile(self, raw: Dict[str, Any]) -> PrinterProfile:
        return PrinterProfile(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", raw.get("id", "printer"))),
            type=str(raw.get("type", "generic")),
            host=str(raw.get("host", "")),
            endpoint=str(raw.get("endpoint", "")),
            api_key=str(raw.get("api_key", "")),
            enabled=bool(raw.get("enabled", True)),
        )

    def list_printers(self) -> Dict[str, Any]:
        db = self._load()
        items: List[Dict[str, Any]] = []
        for p in db.get("printers", []):
            item = dict(p)
            item.pop("api_key", None)
            items.append(item)
        return {"printers": items}

    def get_profile(self, printer_id: str) -> PrinterProfile:
        db = self._load()
        raw = next((p for p in db.get("printers", []) if p.get("id") == printer_id), None)
        if raw is None:
            raise KeyError(f"Printer '{printer_id}' not found")
        return self._to_profile(raw)

    def _driver_for(self, profile: PrinterProfile) -> PrinterDriver:
        return self._drivers.get(profile.type.lower(), self._drivers["generic"])

    def send_3d_print(self, printer_id: str, file_path: str) -> Dict[str, Any]:
        profile = self.get_profile(printer_id)
        if not profile.enabled:
            raise RuntimeError(f"Printer '{printer_id}' is disabled")

        gcode = Path(file_path)
        if not gcode.is_absolute():
            gcode = (self._workspace_root / gcode).resolve()

        driver = self._driver_for(profile)
        return driver.send_3d_print(profile, gcode)

    def get_printer_status(self, printer_id: str) -> Dict[str, Any]:
        profile = self.get_profile(printer_id)
        if not profile.enabled:
            return {
                "success": True,
                "printer_id": printer_id,
                "state": "disabled",
                "details": {},
            }
        driver = self._driver_for(profile)
        return driver.get_printer_status(profile)

    def upsert_printer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        db = self._load()
        pid = str(payload.get("id") or "").strip() or f"printer_{len(db.get('printers', [])) + 1}"
        profile = PrinterProfile(
            id=pid,
            name=str(payload.get("name") or pid),
            type=str(payload.get("type") or "generic"),
            host=str(payload.get("host") or ""),
            endpoint=str(payload.get("endpoint") or ""),
            api_key=str(payload.get("api_key") or ""),
            enabled=bool(payload.get("enabled", True)),
        )

        items = db.get("printers", [])
        existing = next((p for p in items if p.get("id") == pid), None)
        serialized = asdict(profile)
        if existing:
            existing.update(serialized)
        else:
            items.append(serialized)
        db["printers"] = items
        self._save(db)
        redacted = dict(serialized)
        redacted.pop("api_key", None)
        return redacted
