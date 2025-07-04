#!/usr/bin/env python3
"""
Test script to verify model cache list is complete
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_model_list_completeness():
    """Test if all models used in the application are in the cache list."""
    try:
        from utils.model_cache import list_cached_flux_models
        
        # Get the current model list
        model_info = list_cached_flux_models()
        model_names = list(model_info.keys())
        
        print("✅ Current model list updated successfully")
        print("\n📋 Complete Individual Models List:")
        print("=" * 60)
        
        # Group models by category
        core_models = [m for m in model_names if "FLUX.1-" in m and "-lora" not in m]
        lora_models = [m for m in model_names if "-lora" in m]
        utility_models = [m for m in model_names if "FLUX" not in m]
        
        print("🎯 Core FLUX Models:")
        for model in sorted(core_models):
            print(f"   • {model}")
        
        print("\n🎨 LoRA Models:")
        for model in sorted(lora_models):
            print(f"   • {model}")
        
        print("\n🛠️ Utility Models:")
        for model in sorted(utility_models):
            print(f"   • {model}")
        
        print(f"\n📊 Total: {len(model_names)} models")
        print("=" * 60)
        
        # Verify expected models are present
        expected_models = {
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-Fill-dev",
            "black-forest-labs/FLUX.1-Kontext-dev",
            "black-forest-labs/FLUX.1-Redux-dev",
            "black-forest-labs/FLUX.1-Depth-dev-lora",
            "black-forest-labs/FLUX.1-Canny-dev-lora",
            "LiheYoung/depth-anything-large-hf",
            "briaai/RMBG-2.0"
        }
        
        missing_models = expected_models - set(model_names)
        extra_models = set(model_names) - expected_models
        
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            return False
        
        if extra_models:
            print(f"ℹ️ Extra models found: {extra_models}")
        
        print("✅ All expected models are present in the cache list!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🔄 Testing model cache list completeness...")
    print()
    
    if test_model_list_completeness():
        print("\n🎉 Model cache list is complete and up to date!")
        return True
    else:
        print("\n❌ Model cache list needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)