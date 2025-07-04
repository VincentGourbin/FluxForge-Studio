#!/usr/bin/env python3
"""
Quick cache check script for FluxForge Studio.
Verifies that all required models are cached and ready to use.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    from utils.model_cache import get_cache_stats
    
    print("🔍 Quick Cache Check for FluxForge Studio")
    print("=" * 40)
    
    stats = get_cache_stats()
    
    if stats['cache_percentage'] == 100.0:
        print("🎉 ALL MODELS CACHED! Ready to use offline.")
        print(f"📊 Total: {stats['cached_models']}/{stats['total_models']} models")
        print(f"💾 Size: {stats['total_size_formatted']}")
        print("✅ No downloads needed - everything is ready!")
    else:
        print(f"⚠️  {stats['cached_models']}/{stats['total_models']} models cached ({stats['cache_percentage']:.1f}%)")
        print("📥 Missing models:")
        for model_name, info in stats["models"].items():
            if not info["cached"]:
                print(f"   ❌ {model_name}")
        print("\n💡 Run: python optimize_cache.py --predownload")
    
    print("=" * 40)

if __name__ == "__main__":
    main()