#!/usr/bin/env python3
"""
Show potential cleanup savings without actually removing files.
"""

import os
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
    
    # Models that can be removed (replaced by LoRA versions)
    obsolete_models = [
        "models--black-forest-labs--FLUX.1-Canny-dev"  # Replaced by LoRA version
    ]
    
    print("💾 POTENTIAL CLEANUP SAVINGS")
    print("=" * 50)
    print("These standalone models can be safely removed:")
    print("(They have been replaced by more efficient LoRA versions)")
    print()
    
    total_savings = 0
    
    for model_dir_name in obsolete_models:
        model_path = cache_dir / model_dir_name
        
        if model_path.exists():
            size = get_directory_size(model_path)
            size_formatted = format_size(size)
            
            print(f"📦 {model_dir_name}")
            print(f"   Size: {size_formatted}")
            print(f"   Replaced by: LoRA version (1.2GB)")
            print(f"   Can save: {size_formatted}")
            total_savings += size
        else:
            print(f"📦 {model_dir_name}: Already removed ✅")
        
        print()
    
    if total_savings > 0:
        print(f"🎯 TOTAL POTENTIAL SAVINGS: {format_size(total_savings)}")
        print()
        print("💡 To clean up obsolete models:")
        print("   python cleanup_obsolete_models.py")
        print()
        print("✅ After cleanup, you'll have the same functionality")
        print("   using efficient LoRA versions instead of standalone models")
    else:
        print("✅ No obsolete models found - cache is already optimized")
    
    print("=" * 50)

if __name__ == "__main__":
    main()