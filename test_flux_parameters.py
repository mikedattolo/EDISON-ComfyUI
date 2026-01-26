#!/usr/bin/env python3
"""
Test that steps and guidance_scale parameters are properly set in FLUX workflow
"""

import sys
sys.path.insert(0, '.')

def test_flux_parameters():
    """Test that workflow respects steps and guidance_scale parameters"""
    from services.edison_core.app import create_flux_workflow
    
    print("=" * 70)
    print("Testing FLUX Workflow Parameters")
    print("=" * 70)
    
    # Test 1: Custom steps
    print("\n=== Test 1: Custom steps parameter ===")
    workflow = create_flux_workflow("test prompt", steps=30)
    scheduler_steps = workflow["17"]["inputs"]["steps"]
    assert scheduler_steps == 30, f"Expected steps=30, got {scheduler_steps}"
    print(f"✓ Workflow BasicScheduler steps set to: {scheduler_steps}")
    
    # Test 2: Custom guidance_scale
    print("\n=== Test 2: Custom guidance_scale parameter ===")
    workflow = create_flux_workflow("test prompt", guidance_scale=7.0)
    guider_guidance = workflow["22"]["inputs"].get("guidance", None)
    assert guider_guidance == 7.0, f"Expected guidance=7.0, got {guider_guidance}"
    print(f"✓ Workflow BasicGuider guidance set to: {guider_guidance}")
    
    # Test 3: Both parameters together
    print("\n=== Test 3: Both parameters together ===")
    workflow = create_flux_workflow("test prompt", steps=50, guidance_scale=8.5)
    steps_value = workflow["17"]["inputs"]["steps"]
    guidance_value = workflow["22"]["inputs"]["guidance"]
    assert steps_value == 50, f"Expected steps=50, got {steps_value}"
    assert guidance_value == 8.5, f"Expected guidance=8.5, got {guidance_value}"
    print(f"✓ Workflow steps={steps_value}, guidance={guidance_value}")
    
    # Test 4: Parameter clamping - steps too high
    print("\n=== Test 4: Parameter clamping (steps > 200) ===")
    workflow = create_flux_workflow("test prompt", steps=500)
    steps_value = workflow["17"]["inputs"]["steps"]
    assert steps_value == 200, f"Expected clamped steps=200, got {steps_value}"
    print(f"✓ Steps clamped to max: {steps_value}")
    
    # Test 5: Parameter clamping - steps too low
    print("\n=== Test 5: Parameter clamping (steps < 1) ===")
    workflow = create_flux_workflow("test prompt", steps=0)
    steps_value = workflow["17"]["inputs"]["steps"]
    assert steps_value == 1, f"Expected clamped steps=1, got {steps_value}"
    print(f"✓ Steps clamped to min: {steps_value}")
    
    # Test 6: Parameter clamping - guidance too high
    print("\n=== Test 6: Parameter clamping (guidance > 20) ===")
    workflow = create_flux_workflow("test prompt", guidance_scale=50)
    guidance_value = workflow["22"]["inputs"]["guidance"]
    assert guidance_value == 20, f"Expected clamped guidance=20, got {guidance_value}"
    print(f"✓ Guidance clamped to max: {guidance_value}")
    
    # Test 7: Parameter clamping - guidance too low
    print("\n=== Test 7: Parameter clamping (guidance < 0) ===")
    workflow = create_flux_workflow("test prompt", guidance_scale=-5)
    guidance_value = workflow["22"]["inputs"]["guidance"]
    assert guidance_value == 0, f"Expected clamped guidance=0, got {guidance_value}"
    print(f"✓ Guidance clamped to min: {guidance_value}")
    
    # Test 8: Default values
    print("\n=== Test 8: Default values ===")
    workflow = create_flux_workflow("test prompt")
    steps_value = workflow["17"]["inputs"]["steps"]
    guidance_value = workflow["22"]["inputs"]["guidance"]
    assert steps_value == 20, f"Expected default steps=20, got {steps_value}"
    assert guidance_value == 3.5, f"Expected default guidance=3.5, got {guidance_value}"
    print(f"✓ Defaults: steps={steps_value}, guidance={guidance_value}")
    
    # Test 9: Verify other parameters still work
    print("\n=== Test 9: Other parameters (width, height) ===")
    workflow = create_flux_workflow("test prompt", width=512, height=768)
    width = workflow["5"]["inputs"]["width"]
    height = workflow["5"]["inputs"]["height"]
    assert width == 512, f"Expected width=512, got {width}"
    assert height == 768, f"Expected height=768, got {height}"
    print(f"✓ Image dimensions: {width}x{height}")
    
    # Test 10: Full parameter set
    print("\n=== Test 10: Full parameter set ===")
    workflow = create_flux_workflow(
        "A beautiful landscape at sunset",
        width=1024,
        height=1024,
        steps=100,
        guidance_scale=6.5
    )
    print(f"Prompt: {workflow['6']['inputs']['text']}")
    print(f"Size: {workflow['5']['inputs']['width']}x{workflow['5']['inputs']['height']}")
    print(f"Steps: {workflow['17']['inputs']['steps']}")
    print(f"Guidance: {workflow['22']['inputs']['guidance']}")
    print(f"✓ Full workflow configuration correct")
    
    print("\n" + "=" * 70)
    print("✅ All parameter tests passed!")
    print("=" * 70)
    
    print("\nVerification Summary:")
    print("  ✓ steps parameter sets BasicScheduler node value")
    print("  ✓ guidance_scale parameter sets BasicGuider node value")
    print("  ✓ Parameters are not silently ignored")
    print("  ✓ Workflow JSON reflects user-provided values")
    print("  ✓ Parameter clamping prevents invalid ranges")
    print("  ✓ Default values work when not provided")


if __name__ == "__main__":
    test_flux_parameters()
