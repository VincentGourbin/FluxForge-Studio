#!/usr/bin/env python3
"""
Cleanup Obsolete Models Script

Removes large standalone models that have been replaced by more efficient LoRA versions.
This script specifically removes:
- black-forest-labs/FLUX.1-Canny-dev (replaced by FLUX.1-Canny-dev-lora)
- Any other standalone models that are no longer needed

This can free up significant disk space (40+ GB).
"""

import os
import shutil
from pathlib import Path

def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    default_cache = Path.home() / ".cache" / "huggingface" / "hub"
    return Path(os.environ.get("HF_HOME", default_cache.parent)) / "hub"

def format_size(size_bytes):
    """Format size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def get_directory_size(path):
    """Calculate directory size."""
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size

def main():
    cache_dir = get_hf_cache_dir()
    
    # Models to remove (replaced by LoRA versions)
    obsolete_models = [
        "models--black-forest-labs--FLUX.1-Canny-dev"  # Replaced by LoRA version
    ]
    
    print("🧹 OBSOLETE MODEL CLEANUP")
    print("=" * 50)
    print("This will remove standalone models that have been replaced by LoRA versions.")
    print("LoRA versions are more efficient and provide the same functionality.")
    print()
    
    total_freed = 0
    
    for model_dir_name in obsolete_models:
        model_path = cache_dir / model_dir_name
        
        if model_path.exists():
            size = get_directory_size(model_path)
            size_formatted = format_size(size)
            
            print(f"📦 Found: {model_dir_name}")
            print(f"   Size: {size_formatted}")
            
            # Ask for confirmation
            response = input(f"   Remove this model? (y/N): ").lower().strip()
            
            if response in ['y', 'yes']:
                try:
                    shutil.rmtree(model_path)
                    print(f"   ✅ Removed successfully")
                    total_freed += size
                except Exception as e:
                    print(f"   ❌ Failed to remove: {e}")
            else:
                print(f"   ⏭️ Skipped")
        else:
            print(f"📦 {model_dir_name}: Not found (already clean)")
        
        print()
    
    if total_freed > 0:
        print(f"🎉 Cleanup completed!")
        print(f"💾 Total space freed: {format_size(total_freed)}")
        print("✅ Your cache is now optimized for LoRA-based models")
    else:
        print("✅ No obsolete models found - cache is already optimized")
    
    print("=" * 50)

if __name__ == "__main__":
    main()