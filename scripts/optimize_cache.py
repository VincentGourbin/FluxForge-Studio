#!/usr/bin/env python3
"""
Cache Optimization Script

Script to help optimize model downloads and cache usage for MFLUX-Gradio.
Can be run to pre-download models or check cache status.

Usage:
    python scripts/optimize_cache.py --status          # Show cache status
    python scripts/optimize_cache.py --predownload     # Pre-download missing models
    python scripts/optimize_cache.py --help            # Show help
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def main():
    parser = argparse.ArgumentParser(description="FLUX Models Cache Optimization")
    parser.add_argument("--status", action="store_true", help="Show cache status")
    parser.add_argument("--predownload", action="store_true", help="Pre-download missing models")
    parser.add_argument("--model", type=str, help="Specific model to download")
    
    args = parser.parse_args()
    
    if args.status:
        from utils.model_cache import print_cache_status
        print_cache_status()
        
    elif args.predownload:
        from utils.model_cache import list_cached_flux_models, pre_download_model
        
        print("🚀 Starting pre-download of missing FLUX models...")
        models = list_cached_flux_models()
        
        for model_name, info in models.items():
            if not info["cached"]:
                print(f"\n📥 Pre-downloading {model_name}...")
                success = pre_download_model(model_name)
                if success:
                    print(f"✅ {model_name} downloaded successfully")
                else:
                    print(f"❌ Failed to download {model_name}")
            else:
                print(f"✅ {model_name} already cached ({info['size_formatted']})")
        
        print("\n🎉 Pre-download completed!")
        
    elif args.model:
        from utils.model_cache import pre_download_model
        print(f"📥 Downloading specific model: {args.model}")
        success = pre_download_model(args.model)
        if success:
            print(f"✅ {args.model} downloaded successfully")
        else:
            print(f"❌ Failed to download {args.model}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()