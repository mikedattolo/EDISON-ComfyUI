"""Model hot-swap manager (optional)."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, object] = {}
        self.model_cache: Dict[str, object] = {}

    def load_model(self, model_name: str, model_path: str, n_ctx: int, tensor_split: Optional[List[float]] = None):
        """Load a model into GPU memory or reuse a cached instance."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        if model_name in self.model_cache:
            model = self.model_cache[model_name]
            self.loaded_models[model_name] = model
            logger.info(f"Loaded {model_name} from RAM cache")
            return model

        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(f"llama-cpp-python not installed: {e}")

        logger.info(f"Loading {model_name} from disk")
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            tensor_split=tensor_split or [0.5, 0.25, 0.25],
            verbose=True
        )
        self.loaded_models[model_name] = model
        return model

    def unload_model(self, model_name: str):
        """Unload a model from GPU while keeping a cached reference."""
        model = self.loaded_models.pop(model_name, None)
        if not model:
            return
        self.model_cache[model_name] = model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(f"Unloaded {model_name} from GPU")
