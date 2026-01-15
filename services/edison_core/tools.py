"""
ComfyUI Integration Tools
"""

import requests
import json
from typing import Dict, Any

class ComfyUITools:
    """Tools for interacting with ComfyUI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config["host"]
        self.port = config["port"]
        self.base_url = f"http://{self.host}:{self.port}"
    
    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a ComfyUI workflow"""
        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get the current queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def interrupt_execution(self) -> Dict[str, Any]:
        """Interrupt current execution"""
        try:
            response = requests.post(f"{self.base_url}/interrupt")
            response.raise_for_status()
            return {"status": "interrupted"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
