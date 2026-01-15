"""
Prompt Building and Management
"""

from typing import List, Dict

class PromptBuilder:
    """Build prompts for AI interactions"""
    
    SYSTEM_PROMPT = """You are EDISON, an AI assistant specialized in ComfyUI workflow creation and image generation.
You have access to ComfyUI tools and can help users create, modify, and execute workflows.
Always be helpful, precise, and creative in your responses."""
    
    def __init__(self):
        self.system_prompt = self.SYSTEM_PROMPT
    
    def build(self, user_message: str, context: List[str] = None) -> str:
        """Build a complete prompt with context"""
        prompt_parts = [self.system_prompt]
        
        if context:
            context_str = "\n\n".join(context)
            prompt_parts.append(f"\nRelevant Context:\n{context_str}")
        
        prompt_parts.append(f"\nUser: {user_message}")
        
        return "\n".join(prompt_parts)
    
    def build_workflow_prompt(self, intent: str, parameters: Dict) -> str:
        """Build a prompt for workflow generation"""
        param_str = "\n".join([f"- {k}: {v}" for k, v in parameters.items()])
        
        return f"""Generate a ComfyUI workflow for the following intent:

Intent: {intent}

Parameters:
{param_str}

Return a valid ComfyUI workflow JSON."""
    
    def build_analysis_prompt(self, image_path: str, question: str) -> str:
        """Build a prompt for image analysis"""
        return f"""Analyze the image at: {image_path}

Question: {question}

Provide a detailed analysis."""
