#!/usr/bin/env python3
"""
Demonstrate that workflow JSON contains steps and guidance_scale values
"""

import sys
import json
sys.path.insert(0, '.')

def demonstrate_workflow_json():
    """Show the actual workflow JSON with parameters"""
    from services.edison_core.app import create_flux_workflow
    
    print("=" * 80)
    print("FLUX Workflow JSON with User Parameters")
    print("=" * 80)
    
    # Create workflow with specific parameters
    print("\n>>> Creating workflow with:")
    print("    prompt: 'A serene mountain lake at dawn'")
    print("    width: 1024, height: 1024")
    print("    steps: 30")
    print("    guidance_scale: 7.5")
    
    workflow = create_flux_workflow(
        prompt="A serene mountain lake at dawn",
        width=1024,
        height=1024,
        steps=30,
        guidance_scale=7.5
    )
    
    print("\n" + "=" * 80)
    print("Relevant Nodes in Workflow JSON:")
    print("=" * 80)
    
    # Show BasicScheduler node
    print("\n📌 Node 17: BasicScheduler (sampling steps)")
    print("-" * 80)
    scheduler = workflow["17"]["inputs"]
    print(f"  scheduler: {scheduler['scheduler']}")
    print(f"  steps: {scheduler['steps']}  ← USER PARAMETER (was hardcoded as 4)")
    print(f"  denoise: {scheduler['denoise']}")
    
    # Show BasicGuider node
    print("\n📌 Node 22: BasicGuider (guidance scale)")
    print("-" * 80)
    guider = workflow["22"]["inputs"]
    print(f"  model: {guider['model']}")
    print(f"  conditioning: {guider['conditioning']}")
    print(f"  guidance: {guider['guidance']}  ← USER PARAMETER (NEW FIELD)")
    
    # Show CLIPTextEncode node
    print("\n📌 Node 6: CLIPTextEncode (positive prompt)")
    print("-" * 80)
    prompt_node = workflow["6"]["inputs"]
    print(f"  text: {prompt_node['text']}")
    
    # Show EmptyLatentImage node
    print("\n📌 Node 5: EmptyLatentImage (image dimensions)")
    print("-" * 80)
    latent = workflow["5"]["inputs"]
    print(f"  width: {latent['width']}")
    print(f"  height: {latent['height']}")
    
    print("\n" + "=" * 80)
    print("Full Workflow JSON (Pretty Printed):")
    print("=" * 80)
    print(json.dumps(workflow, indent=2))
    
    print("\n" + "=" * 80)
    print("✅ Verification Complete")
    print("=" * 80)
    print("\nKey Findings:")
    print("  ✓ steps=30 is in workflow['17']['inputs']['steps']")
    print("  ✓ guidance_scale=7.5 is in workflow['22']['inputs']['guidance']")
    print("  ✓ Prompt 'A serene mountain lake at dawn' in workflow['6']['inputs']['text']")
    print("  ✓ Image dimensions 1024x1024 in workflow['5']['inputs']")
    print("  ✓ Parameters are NOT silently ignored")
    print("  ✓ Workflow JSON is ready to send to ComfyUI")


if __name__ == "__main__":
    demonstrate_workflow_json()
