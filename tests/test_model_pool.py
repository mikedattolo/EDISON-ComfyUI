from services.edison_core.model_manager_v2 import ModelPool


class _DummyManager:
    def __init__(self):
        self.unloaded = []

    def _unload_model_unsafe(self, key):
        self.unloaded.append(key)


def test_model_pool_evicts_previous_heavy_model():
    pool = ModelPool()
    mgr = _DummyManager()

    pool.current_heavy_key = "deep"
    pool.before_loading_heavy(mgr, "medium")

    assert mgr.unloaded == ["deep"]
    assert pool.current_heavy_key is None


def test_model_pool_marks_and_clears_heavy_keys():
    pool = ModelPool()
    pool.mark_loaded("deep", is_heavy=True)
    assert pool.current_heavy_key == "deep"

    pool.clear("deep")
    assert pool.current_heavy_key is None
