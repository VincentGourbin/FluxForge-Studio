#!/usr/bin/env python3
"""
Complete cache summary for FluxForge Studio.
Shows current status, optimizations, and recommendations.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    from utils.model_cache import get_cache_stats
    import os
    
    print("🗂️ FLUXFORGE STUDIO CACHE SUMMARY")
    print("=" * 60)
    
    # Current cache status
    stats = get_cache_stats()
    
    print("📊 CURRENT STATUS:")
    print(f"   Models cached: {stats['cached_models']}/{stats['total_models']} ({stats['cache_percentage']:.1f}%)")
    print(f"   Cache size: {stats['total_size_formatted']}")
    print()
    
    if stats['cache_percentage'] == 100.0:
        print("🎉 ALL REQUIRED MODELS CACHED!")
        print("✅ Ready for offline use")
    else:
        print("⚠️  Some models missing:")
        for model_name, info in stats["models"].items():
            if not info["cached"]:
                print(f"   ❌ {model_name}")
        print(f"\n💡 Run: python optimize_cache.py --predownload")
    
    print()
    
    # Check for optimization opportunities
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    obsolete_canny = cache_dir / "models--black-forest-labs--FLUX.1-Canny-dev"
    
    if obsolete_canny.exists():
        print("💾 OPTIMIZATION OPPORTUNITY:")
        print("   Large standalone model can be replaced by LoRA version")
        print(f"   Potential savings: ~81GB")
        print(f"   Run: python show_cleanup_savings.py")
        print()
    
    # Show model breakdown
    print("📋 MODEL BREAKDOWN:")
    for model_name, info in stats["models"].items():
        status = "✅" if info["cached"] else "❌"
        size_info = f"({info['size_formatted']})" if info["cached"] else ""
        
        # Add model type annotation
        if "lora" in model_name.lower():
            model_type = "[LoRA]"
        elif "depth-anything" in model_name.lower():
            model_type = "[Depth]"
        elif "rmbg" in model_name.lower():
            model_type = "[BG Removal]"
        else:
            model_type = "[Core]"
        
        print(f"   {status} {model_name} {model_type} {size_info}")
    
    print()
    print("🚀 OPTIMIZATION BENEFITS:")
    print("   • LoRA models: Same quality, 98% less space")
    print("   • Automatic caching: No re-downloads")
    print("   • Offline capability: Works without internet")
    print("   • Faster loading: Local models load instantly")
    
    print("=" * 60)

if __name__ == "__main__":
    main()