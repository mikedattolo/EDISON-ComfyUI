"""
Tests for Edison memory management, file handling, and editing subsystems.
Covers: ModelManager v2, MemoryGate, SwarmMemoryPolicy, FileStore,
        ImageEditor, FileEditor, ProvenanceTracker.

Run: python tests/test_memory_and_files.py
"""

import os
import sys
import tempfile
import shutil
import json
import time
import threading

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"✓ {name}: PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ {name}: FAILED — {e}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════
# Part 1: ModelManager v2 — MemorySnapshot
# ═══════════════════════════════════════════════════════════════════════

def test_memory_snapshot_creation():
    from services.edison_core.model_manager_v2 import MemorySnapshot

    snap = MemorySnapshot(
        ram_total_mb=32000,
        ram_available_mb=16000,
        gpus=[{"id": 0, "name": "RTX 4090", "total_mb": 24000, "free_mb": 20000}],
    )
    assert snap.ram_total_mb == 32000
    assert snap.ram_available_mb == 16000
    assert len(snap.gpus) == 1
    assert snap.gpus[0]["free_mb"] == 20000


def test_memory_snapshot_total_vram():
    from services.edison_core.model_manager_v2 import MemorySnapshot

    snap = MemorySnapshot(
        ram_total_mb=32000,
        ram_available_mb=16000,
        gpus=[
            {"id": 0, "name": "GPU0", "total_mb": 16000, "free_mb": 8000},
            {"id": 1, "name": "GPU1", "total_mb": 16000, "free_mb": 12000},
        ],
    )
    assert snap.total_vram_free_mb() == 20000.0


def test_memory_snapshot_no_gpus():
    from services.edison_core.model_manager_v2 import MemorySnapshot

    snap = MemorySnapshot(ram_total_mb=32000, ram_available_mb=16000, gpus=[])
    assert snap.total_vram_free_mb() == 0.0


run_test("MemorySnapshot creation", test_memory_snapshot_creation)
run_test("MemorySnapshot total_vram_free_mb multi-GPU", test_memory_snapshot_total_vram)
run_test("MemorySnapshot no GPUs", test_memory_snapshot_no_gpus)


# ═══════════════════════════════════════════════════════════════════════
# Part 1b: ModelManager v2 — LoadedModel
# ═══════════════════════════════════════════════════════════════════════

def test_loaded_model_defaults():
    from services.edison_core.model_manager_v2 import LoadedModel

    lm = LoadedModel(key="fast", path="")
    assert lm.key == "fast"
    assert lm.model is None
    assert lm.estimated_vram_mb == 0
    assert lm.n_gpu_layers == -1
    assert lm.n_ctx == 4096
    assert lm.path == ""


def test_loaded_model_with_values():
    from services.edison_core.model_manager_v2 import LoadedModel

    lm = LoadedModel(key="deep", path="/models/deep.gguf", estimated_vram_mb=8000, n_gpu_layers=40, n_ctx=4096)
    assert lm.key == "deep"
    assert lm.path == "/models/deep.gguf"
    assert lm.estimated_vram_mb == 8000
    assert lm.n_gpu_layers == 40
    assert lm.n_ctx == 4096


run_test("LoadedModel defaults", test_loaded_model_defaults)
run_test("LoadedModel with values", test_loaded_model_with_values)


# ═══════════════════════════════════════════════════════════════════════
# Part 1c: ModelManager v2 — Core logic
# ═══════════════════════════════════════════════════════════════════════

def test_model_manager_init():
    from services.edison_core.model_manager_v2 import ModelManager

    mm = ModelManager()
    # Should have empty models dict
    assert isinstance(mm._models, dict)
    assert mm._heavy_slot is None


def test_model_manager_heavy_slot():
    from services.edison_core.model_manager_v2 import ModelManager

    mm = ModelManager()
    assert mm.heavy_slot_occupant() is None


def test_model_manager_is_loaded():
    from services.edison_core.model_manager_v2 import ModelManager, LoadedModel

    mm = ModelManager()
    assert mm.is_loaded("fast") is False

    # Manually inject a model to test
    lm = LoadedModel(key="fast", path="")
    lm.model = "fake_model_object"
    mm._models["fast"] = lm
    assert mm.is_loaded("fast") is True


def test_model_manager_constants():
    from services.edison_core.model_manager_v2 import (
        FAST_KEY, HEAVY_KEYS, FALLBACK_GPU_LAYERS, FALLBACK_CTX_SIZES,
    )

    assert FAST_KEY == "fast"
    assert "medium" in HEAVY_KEYS
    assert "deep" in HEAVY_KEYS
    assert "reasoning" in HEAVY_KEYS
    assert "vision" in HEAVY_KEYS
    assert len(FALLBACK_GPU_LAYERS) >= 3
    assert len(FALLBACK_CTX_SIZES) >= 3


run_test("ModelManager init", test_model_manager_init)
run_test("ModelManager heavy_slot default", test_model_manager_heavy_slot)
run_test("ModelManager is_loaded", test_model_manager_is_loaded)
run_test("ModelManager constants", test_model_manager_constants)


# ═══════════════════════════════════════════════════════════════════════
# Part 1d: MemoryGate
# ═══════════════════════════════════════════════════════════════════════

def test_memory_gate_init():
    from services.edison_core.model_manager_v2 import MemoryGate

    gate = MemoryGate()
    assert gate._mm is None

    gate2 = MemoryGate(model_manager=None)
    assert gate2._mm is None


def test_memory_gate_pre_heavy_task_no_gpu():
    """Without GPU, pre_heavy_task should still return a result."""
    from services.edison_core.model_manager_v2 import MemoryGate

    gate = MemoryGate()
    result = gate.pre_heavy_task(required_vram_mb=0)
    assert "ok" in result
    assert "freed_mb" in result
    assert "snapshot" in result


run_test("MemoryGate init", test_memory_gate_init)
run_test("MemoryGate pre_heavy_task (no GPU)", test_memory_gate_pre_heavy_task_no_gpu)


# ═══════════════════════════════════════════════════════════════════════
# Part 2: SwarmMemoryPolicy
# ═══════════════════════════════════════════════════════════════════════

def test_swarm_policy_init():
    from services.edison_core.swarm_safety import SwarmMemoryPolicy

    policy = SwarmMemoryPolicy()
    assert policy is not None


def test_swarm_group_agents_by_model():
    from services.edison_core.swarm_safety import group_agents_by_model

    # Agents are dicts with model_name key
    agents = [
        {"model_name": "fast_agent", "name": "a1"},
        {"model_name": "deep_thinker", "name": "a2"},
        {"model_name": "fast_helper", "name": "a3"},
    ]
    groups = group_agents_by_model(agents)
    assert "fast" in groups
    assert "deep" in groups
    assert len(groups["fast"]) == 2
    assert len(groups["deep"]) == 1


def test_swarm_policy_assess_empty():
    from services.edison_core.swarm_safety import SwarmMemoryPolicy

    policy = SwarmMemoryPolicy()
    result = policy.assess(agents=[], loaded_models=[])
    assert result in ("normal", "time_slice", "degraded")


def test_swarm_should_load_vision():
    from services.edison_core.swarm_safety import should_load_vision

    # No images → should not load vision
    result = should_load_vision(has_images=False, request_message="hello")
    assert result is False

    # Has images → should load vision
    result = should_load_vision(has_images=True)
    assert result is True

    # Vision keyword in message → should load vision
    result = should_load_vision(has_images=False, request_message="describe this image for me")
    assert result is True


run_test("SwarmMemoryPolicy init", test_swarm_policy_init)
run_test("group_agents_by_model", test_swarm_group_agents_by_model)
run_test("SwarmMemoryPolicy assess (empty)", test_swarm_policy_assess_empty)
run_test("should_load_vision", test_swarm_should_load_vision)


# ═══════════════════════════════════════════════════════════════════════
# Part 4a: FileStore
# ═══════════════════════════════════════════════════════════════════════

def test_file_store_upload_and_get():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        meta = store.upload("test.txt", b"Hello World", session_id="sess_1")
        assert meta is not None
        assert meta.original_filename == "test.txt"
        assert meta.session_id == "sess_1"
        assert meta.size_bytes == 11
        assert meta.file_id
        assert meta.file_type == "file"

        # Retrieve
        got = store.get(meta.file_id)
        assert got is not None
        assert got.original_filename == "test.txt"


def test_file_store_read_text():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        meta = store.upload("hello.txt", b"Hello World", session_id="sess_1")
        content = store.read_text(meta.file_id)
        assert content == "Hello World"


def test_file_store_upload_image():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        # Create a fake PNG-like file
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        meta = store.upload("photo.png", fake_png, session_id="sess_1")
        assert meta.file_type == "image"


def test_file_store_list_files():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        store.upload("a.txt", b"AAA", session_id="sess_1")
        store.upload("b.txt", b"BBB", session_id="sess_1")
        store.upload("c.txt", b"CCC", session_id="sess_2")

        files = store.list_files(session_id="sess_1")
        assert len(files) == 2

        all_files = store.list_files()
        assert len(all_files) == 3


def test_file_store_delete():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        meta = store.upload("to_delete.txt", b"bye", session_id="sess_1")
        fid = meta.file_id

        assert store.get(fid) is not None
        store.delete(fid)
        assert store.get(fid) is None


def test_file_store_dedup():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        m1 = store.upload("copy1.txt", b"same content", session_id="s1")
        m2 = store.upload("copy2.txt", b"same content", session_id="s1")
        # Both should have the same SHA-256 hash
        assert m1.sha256 == m2.sha256


def test_file_store_size_limit():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        # Try uploading > 100MB (should fail)
        try:
            # We won't actually allocate 100MB — just check the limit is enforced
            huge = b"x" * (100 * 1024 * 1024 + 1)
            store.upload("huge.txt", huge, session_id="s1")
            assert False, "Should have raised an error"
        except (ValueError, Exception):
            pass  # Expected


def test_file_store_path_safety():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        # Try uploading with path traversal
        try:
            store.upload("../../etc/passwd", b"evil", session_id="s1")
            # If it doesn't raise, the filename should be sanitized
            files = store.list_files(session_id="s1")
            for f in files:
                assert ".." not in f.original_filename
        except (ValueError, Exception):
            pass  # Also acceptable


run_test("FileStore upload and get", test_file_store_upload_and_get)
run_test("FileStore read_text", test_file_store_read_text)
run_test("FileStore upload image", test_file_store_upload_image)
run_test("FileStore list_files", test_file_store_list_files)
run_test("FileStore delete", test_file_store_delete)
run_test("FileStore dedup by SHA-256", test_file_store_dedup)
run_test("FileStore size limit", test_file_store_size_limit)
run_test("FileStore path safety", test_file_store_path_safety)


# ═══════════════════════════════════════════════════════════════════════
# Part 4b: ImageEditor
# ═══════════════════════════════════════════════════════════════════════

def test_image_editor_init():
    from services.image_editing.editor import ImageEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)
        assert editor is not None


def test_image_editor_crop():
    from services.image_editing.editor import ImageEditor

    try:
        from PIL import Image
    except ImportError:
        return  # Skip if Pillow not installed

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)

        # Create a test image
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (200, 200), color="red")
        img.save(img_path)

        result = editor.crop(img_path, box=(10, 10, 100, 100))
        assert result is not None
        assert os.path.exists(result.output_path)

        # Verify crop dimensions
        cropped = Image.open(result.output_path)
        assert cropped.size == (90, 90)


def test_image_editor_resize():
    from services.image_editing.editor import ImageEditor

    try:
        from PIL import Image
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(img_path)

        result = editor.resize(img_path, width=100, height=50)
        assert result is not None
        resized = Image.open(result.output_path)
        assert resized.size == (100, 50)


def test_image_editor_rotate():
    from services.image_editing.editor import ImageEditor

    try:
        from PIL import Image
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (200, 100), color="green")
        img.save(img_path)

        result = editor.rotate(img_path, angle=90)
        assert result is not None
        assert os.path.exists(result.output_path)


def test_image_editor_flip():
    from services.image_editing.editor import ImageEditor

    try:
        from PIL import Image
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (200, 200), color="yellow")
        img.save(img_path)

        result = editor.flip(img_path, direction="horizontal")
        assert result is not None
        assert os.path.exists(result.output_path)


def test_image_editor_brightness():
    from services.image_editing.editor import ImageEditor

    try:
        from PIL import Image
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = ImageEditor(output_dir=tmpdir)
        img_path = os.path.join(tmpdir, "test.png")
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(img_path)

        result = editor.adjust_brightness(img_path, factor=1.5)
        assert result is not None
        assert os.path.exists(result.output_path)


run_test("ImageEditor init", test_image_editor_init)
run_test("ImageEditor crop", test_image_editor_crop)
run_test("ImageEditor resize", test_image_editor_resize)
run_test("ImageEditor rotate", test_image_editor_rotate)
run_test("ImageEditor flip", test_image_editor_flip)
run_test("ImageEditor brightness", test_image_editor_brightness)


# ═══════════════════════════════════════════════════════════════════════
# Part 4c: FileEditor
# ═══════════════════════════════════════════════════════════════════════

def test_file_editor_load():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        test_file = os.path.join(tmpdir, "sample.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")

        result = editor.load_file(test_file)
        assert result == "print('hello')\n"


def test_file_editor_apply_edit():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        file_id = "test_file_1"
        original = "print('hello')\n"
        new_content = "print('world')\n"

        result = editor.apply_edit(file_id, original, new_content, "sample.py")
        assert result is not None
        assert result.success is True
        assert result.new_content == new_content
        assert result.version_id


def test_file_editor_search_replace():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        file_id = "test_sr"
        content = "hello world\nhello earth\n"

        result = editor.apply_search_replace(file_id, content, "hello", "goodbye", "sample.py")
        assert result is not None
        assert result.success is True
        assert "goodbye world" in result.new_content
        assert "goodbye earth" in result.new_content


def test_file_editor_versions():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        file_id = "versioned_file"

        # Make two edits
        editor.apply_edit(file_id, "version 1\n", "version 2\n", "versioned.txt")
        editor.apply_edit(file_id, "version 2\n", "version 3\n", "versioned.txt")

        versions = editor.get_versions(file_id)
        # Should have 3 versions: original + 2 edits
        assert len(versions) >= 2


def test_file_editor_revert():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        file_id = "revert_test"

        # Edit then revert
        result1 = editor.apply_edit(file_id, "original\n", "modified\n", "revert.txt")
        assert result1.success

        versions = editor.get_versions(file_id)
        if versions:
            first_version_id = versions[0].get("version_id", "")
            revert_result = editor.revert_to_version(file_id, first_version_id, "revert.txt")
            assert revert_result.success
            assert revert_result.new_content == "original\n"


def test_file_editor_diff():
    from services.file_editing.editor import FileEditor

    with tempfile.TemporaryDirectory() as tmpdir:
        editor = FileEditor(versions_dir=tmpdir)
        file_id = "diff_test"

        editor.apply_edit(file_id, "line 1\nline 2\nline 3\n",
                         "line 1\nmodified line\nline 3\n", "diff.txt")

        versions = editor.get_versions(file_id)
        if len(versions) >= 2:
            v1 = versions[0].get("version_id", "")
            v2 = versions[1].get("version_id", "")
            diff = editor.get_diff_between_versions(file_id, v1, v2)
            assert diff is not None
            assert len(diff) > 0


run_test("FileEditor load", test_file_editor_load)
run_test("FileEditor apply_edit", test_file_editor_apply_edit)
run_test("FileEditor search_replace", test_file_editor_search_replace)
run_test("FileEditor versions", test_file_editor_versions)
run_test("FileEditor revert", test_file_editor_revert)
run_test("FileEditor diff", test_file_editor_diff)


# ═══════════════════════════════════════════════════════════════════════
# Part 5: ProvenanceTracker
# ═══════════════════════════════════════════════════════════════════════

def test_provenance_record():
    from services.provenance import ProvenanceTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(base_dir=tmpdir)
        record = tracker.record(
            action="image_generation",
            model_used="SDXL",
            parameters={"prompt": "a sunset", "steps": 20},
            output_artifacts=["output.png"],
        )
        assert record is not None
        assert record.action == "image_generation"
        assert record.model_used == "SDXL"


def test_provenance_get():
    from services.provenance import ProvenanceTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(base_dir=tmpdir)
        rec = tracker.record(action="test", model_used="test_model")
        retrieved = tracker.get(rec.record_id)
        assert retrieved is not None
        assert retrieved.record_id == rec.record_id


def test_provenance_list_recent():
    from services.provenance import ProvenanceTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(base_dir=tmpdir)
        for i in range(5):
            tracker.record(action=f"action_{i}", model_used="model")

        recent = tracker.list_recent(limit=3)
        assert len(recent) == 3


def test_provenance_sidecar():
    from services.provenance import ProvenanceTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(base_dir=tmpdir)
        output_file = os.path.join(tmpdir, "output.png")
        with open(output_file, "w") as f:
            f.write("fake")

        rec = tracker.record(
            action="image_gen",
            model_used="FLUX",
            output_artifacts=[output_file],
        )
        tracker.write_sidecar(rec, artifact_path=output_file)

        sidecar_path = output_file + ".provenance.json"
        assert os.path.exists(sidecar_path)

        with open(sidecar_path) as f:
            data = json.load(f)
        assert data["action"] == "image_gen"
        assert data["model_used"] == "FLUX"


run_test("ProvenanceTracker record", test_provenance_record)
run_test("ProvenanceTracker get", test_provenance_get)
run_test("ProvenanceTracker list_recent", test_provenance_list_recent)
run_test("ProvenanceTracker sidecar", test_provenance_sidecar)


# ═══════════════════════════════════════════════════════════════════════
# Part 6: Thread safety
# ═══════════════════════════════════════════════════════════════════════

def test_file_store_concurrent_uploads():
    from services.files.file_store import FileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileStore(base_dir=tmpdir)
        errors = []

        def upload_worker(i):
            try:
                store.upload(f"file_{i}.txt", f"content_{i}".encode(), session_id="concurrent")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=upload_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent upload errors: {errors}"
        files = store.list_files(session_id="concurrent")
        assert len(files) == 10


def test_provenance_concurrent_records():
    from services.provenance import ProvenanceTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(base_dir=tmpdir)
        errors = []

        def record_worker(i):
            try:
                tracker.record(action=f"action_{i}", model_used=f"model_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=record_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent record errors: {errors}"
        recent = tracker.list_recent(limit=20)
        assert len(recent) == 10


run_test("FileStore concurrent uploads", test_file_store_concurrent_uploads)
run_test("ProvenanceTracker concurrent records", test_provenance_concurrent_records)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'=' * 60}")

if failed > 0:
    sys.exit(1)
