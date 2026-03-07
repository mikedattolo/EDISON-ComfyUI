# 3D Printer Integration

EDISON includes a printer abstraction in `services/edison_core/printer.py` with:
- `PrinterManager`
- `BambuLabDriver`
- `OctoPrintDriver`
- `GenericStubDriver` (placeholder for other vendors)

## Supported Operations
- `list_printers()`
- `send_3d_print(printer_id, file_path)`
- `get_printer_status(printer_id)`

## API Endpoints
- `GET /printing/printers`
- `POST /printing/printers`
- `GET /printing/printers/{printer_id}/status`
- `POST /printing/printers/send`
- `POST /printing/slice-and-send`

All new endpoints return a `success` field and structured error details on failure.

## Bambu LAN Notes
`BambuLabDriver` uses best-effort HTTP endpoints for Bambu Connect/LAN bridge deployments:
- `POST /api/v1/print/jobs` (primary)
- `POST /upload` (fallback)
- `GET /api/v1/printer/status` (primary)
- `GET /status` (fallback)

Configure each printer with `host` or full `endpoint` and optional `api_key`.

## UI
The Printing panel (web UI) now supports loading printers and querying live status for selected devices.
