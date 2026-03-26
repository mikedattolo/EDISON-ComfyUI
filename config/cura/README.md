# CuraEngine Configuration

Place CuraEngine machine definition and profile files in this folder if you want Edison to use CuraEngine directly.

Expected default paths from config/edison.yaml:
- definitions/fdmprinter.def.json
- profiles/standard.inst.cfg

You can change those paths in config/edison.yaml under:
- edison.printing.slicer.cura_definition
- edison.printing.slicer.cura_profile

Notes:
- Paths may be absolute or repository-relative.
- If CuraEngine is installed but the definition file is missing, Edison will detect it but fall back to the next usable slicer.
- PrusaSlicer, OrcaSlicer, and Slic3r remain valid fallbacks.
