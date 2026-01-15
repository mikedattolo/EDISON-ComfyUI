"""
EDISON Chat Node for ComfyUI
Provides AI chat interface within ComfyUI workflows - connects to edison-core service
"""

import requests
import json
import traceback

class EdisonChatNode:
    """
    EDISON Chat Node - Interact with EDISON AI from ComfyUI
    Connects to edison-core FastAPI service at http://127.0.0.1:8811
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! Can you help me create an image?"
                }),
                "mode": (["auto", "chat", "reasoning", "agent", "code"], {
                    "default": "auto"
                }),
                "remember": ("BOOLEAN", {
                    "default": True
                }),
                "timeout_seconds": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600,
                    "step": 10
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "EDISON"
    DESCRIPTION = "Chat with EDISON AI - supports multiple modes (auto/chat/reasoning/agent/code)"
    
    def chat(self, text: str, mode: str = "auto", remember: bool = True, timeout_seconds: int = 120):
        """Send a message to EDISON and get a response"""
        
        # Validate input
        if not text or not text.strip():
            return ("Error: Empty message provided",)
        
        edison_url = "http://127.0.0.1:8811/chat"
        
        try:
            # Prepare request payload
            payload = {
                "message": text.strip(),
                "mode": mode,
                "remember": remember
            }
            
            # Call EDISON API
            response = requests.post(
                edison_url,
                json=payload,
                timeout=timeout_seconds,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for HTTP errors
            if response.status_code == 503:
                error_detail = response.json().get("detail", "Service unavailable")
                return (f"⚠ EDISON Core unavailable: {error_detail}\n\nPlease ensure:\n1. EDISON models are installed in models/llm/\n2. edison-core service is running: sudo systemctl status edison-core",)
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            response_text = result.get("response", "No response from EDISON")
            mode_used = result.get("mode_used", mode)
            model_used = result.get("model_used", "unknown")
            
            # Format response with metadata
            full_response = f"{response_text}\n\n[Mode: {mode_used} | Model: {model_used}]"
            
            return (full_response,)
        
        except requests.exceptions.Timeout:
            error_msg = f"⏱ EDISON request timed out after {timeout_seconds}s\n\nTry:\n- Increasing timeout_seconds\n- Using a faster mode (chat instead of reasoning)\n- Checking if edison-core service is responding"
            return (error_msg,)
        
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to EDISON Core at http://127.0.0.1:8811\n\nPlease ensure:\n1. edison-core service is running: sudo systemctl start edison-core\n2. Check service logs: journalctl -u edison-core -f\n3. Test health: curl http://127.0.0.1:8811/health"
            return (error_msg,)
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"❌ EDISON API error (HTTP {response.status_code})\n\nDetails: {str(e)}\n\nCheck logs: journalctl -u edison-core -n 50"
            return (error_msg,)
        
        except json.JSONDecodeError:
            error_msg = "❌ Invalid JSON response from EDISON\n\nThe service may be starting up or misconfigured.\nCheck logs: journalctl -u edison-core -n 50"
            return (error_msg,)
        
        except Exception as e:
            # Catch-all for unexpected errors - never crash ComfyUI
            error_msg = f"❌ Unexpected error communicating with EDISON:\n{str(e)}\n\nStack trace:\n{traceback.format_exc()}"
            return (error_msg,)


class EdisonHealthCheck:
    """
    Check EDISON service health
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check_health"
    CATEGORY = "EDISON"
    DESCRIPTION = "Check EDISON core and coral service health"
    
    def check_health(self):
        """Check health of EDISON services"""
        
        status_lines = ["=== EDISON Health Check ===\n"]
        
        # Check coral service
        try:
            coral_resp = requests.get("http://127.0.0.1:8808/health", timeout=5)
            if coral_resp.status_code == 200:
                coral_data = coral_resp.json()
                status_lines.append("✓ edison-coral: HEALTHY")
                status_lines.append(f"  TPU Available: {coral_data.get('tpu_available', 'unknown')}")
                status_lines.append(f"  Method: {coral_data.get('intent_classifier_method', 'unknown')}")
            else:
                status_lines.append(f"⚠ edison-coral: HTTP {coral_resp.status_code}")
        except Exception as e:
            status_lines.append(f"✗ edison-coral: OFFLINE ({str(e)})")
        
        status_lines.append("")
        
        # Check core service
        try:
            core_resp = requests.get("http://127.0.0.1:8811/health", timeout=5)
            if core_resp.status_code == 200:
                core_data = core_resp.json()
                status_lines.append("✓ edison-core: HEALTHY")
                models = core_data.get('models_loaded', {})
                status_lines.append(f"  Fast Model: {'✓' if models.get('fast_model') else '✗'}")
                status_lines.append(f"  Deep Model: {'✓' if models.get('deep_model') else '✗'}")
                status_lines.append(f"  Qdrant: {'✓' if core_data.get('qdrant_ready') else '✗'}")
            else:
                status_lines.append(f"⚠ edison-core: HTTP {core_resp.status_code}")
        except Exception as e:
            status_lines.append(f"✗ edison-core: OFFLINE ({str(e)})")
        
        return ("\n".join(status_lines),)


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "EdisonChatNode": EdisonChatNode,
    "EdisonHealthCheck": EdisonHealthCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdisonChatNode": "EDISON Chat",
    "EdisonHealthCheck": "EDISON Health Check",
}
