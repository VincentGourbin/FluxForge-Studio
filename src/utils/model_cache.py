"""
Model Cache Management Module

Utilities for managing model downloads and caching to avoid repeated downloads
of large models like FLUX.1-Canny-dev, FLUX.1-Depth-dev, etc.

Features:
- Check if models are already cached locally
- Pre-download models for offline usage
- Manage cache directory and cleanup
- Display cache information and statistics

Author: MFLUX Team
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import shutil

def get_hf_cache_dir():
    """
    Get the HuggingFace cache directory path.
    
    Returns:
        Path: Path to HuggingFace cache directory
    """
    default_cache = Path.home() / ".cache" / "huggingface"
    return Path(os.environ.get("HF_HOME", default_cache))

def check_model_cached(model_name):
    """
    Check if a model is already cached locally.
    
    Args:
        model_name (str): Name of the HuggingFace model (e.g., "black-forest-labs/FLUX.1-Canny-dev")
        
    Returns:
        tuple: (is_cached: bool, cache_size: int, cache_path: Path)
    """
    try:
        cache_dir = get_hf_cache_dir() / "hub"
        
        # Convert model name to cache directory format
        cache_model_name = model_name.replace("/", "--")
        model_cache_path = cache_dir / f"models--{cache_model_name}"
        
        if not model_cache_path.exists():
            return False, 0, None
        
        # Check if snapshots directory exists and has content
        snapshots_dir = model_cache_path / "snapshots"
        
        if not snapshots_dir.exists():
            return False, 0, model_cache_path
        
        # Check if there are any snapshot directories
        snapshot_dirs = list(snapshots_dir.iterdir())
        if not snapshot_dirs:
            return False, 0, model_cache_path
        
        # Calculate cache size from all blobs and snapshots
        total_size = 0
        
        # Calculate size from blobs directory
        blobs_dir = model_cache_path / "blobs"
        if blobs_dir.exists():
            for blob_file in blobs_dir.iterdir():
                if blob_file.is_file():
                    total_size += blob_file.stat().st_size
        
        # Also check snapshots for symlinks/files
        for snapshot_dir in snapshot_dirs:
            if snapshot_dir.is_dir():
                for root, dirs, files in os.walk(snapshot_dir):
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.is_file() and not file_path.is_symlink():
                            total_size += file_path.stat().st_size
        
        # Consider model cached if it has some content
        has_content = total_size > 0
        
        return has_content, total_size, model_cache_path
        
    except Exception as e:
        print(f"Error checking cache for {model_name}: {e}")
        return False, 0, None

def format_size(size_bytes):
    """
    Format size in bytes to human readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def list_cached_flux_models():
    """
    List all cached FLUX models and their information.
    
    Returns:
        dict: Dictionary of model info with cache status
    """
    flux_models = [
        # Core FLUX models
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell", 
        "black-forest-labs/FLUX.1-Fill-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "black-forest-labs/FLUX.1-Redux-dev",
        
        # LoRA models (preferred for efficiency)
        "black-forest-labs/FLUX.1-Depth-dev-lora",
        "black-forest-labs/FLUX.1-Canny-dev-lora",
        
        # Other utility models
        "LiheYoung/depth-anything-large-hf",  # Depth map generation
        "briaai/RMBG-2.0"  # Background removal
    ]
    
    model_info = {}
    
    for model in flux_models:
        is_cached, cache_size, cache_path = check_model_cached(model)
        model_info[model] = {
            "cached": is_cached,
            "size": cache_size,
            "size_formatted": format_size(cache_size),
            "path": cache_path
        }
    
    return model_info

def pre_download_model(model_name, force=False):
    """
    Pre-download a model to local cache.
    
    Args:
        model_name (str): Name of the HuggingFace model
        force (bool): Force re-download even if cached
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"📥 Pre-downloading model: {model_name}")
        
        if not force:
            is_cached, _, _ = check_model_cached(model_name)
            if is_cached:
                print(f"✅ Model {model_name} already cached")
                return True
        
        # Use HuggingFace API to get file list
        api = HfApi()
        files = api.list_repo_files(model_name)
        
        # Download key files (config and model weights)
        key_files = [f for f in files if f.endswith(('.json', '.safetensors', '.bin'))]
        
        for file in key_files:
            print(f"📥 Downloading {file}...")
            hf_hub_download(
                repo_id=model_name,
                filename=file,
                force_download=force
            )
        
        print(f"✅ Model {model_name} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def get_cache_stats():
    """
    Get cache statistics for all FLUX models.
    
    Returns:
        dict: Cache statistics
    """
    model_info = list_cached_flux_models()
    
    total_models = len(model_info)
    cached_models = sum(1 for info in model_info.values() if info["cached"])
    total_size = sum(info["size"] for info in model_info.values())
    
    return {
        "total_models": total_models,
        "cached_models": cached_models,
        "cache_percentage": (cached_models / total_models) * 100 if total_models > 0 else 0,
        "total_size": total_size,
        "total_size_formatted": format_size(total_size),
        "models": model_info
    }

def print_cache_status():
    """
    Print detailed cache status for all FLUX models.
    """
    print("=" * 60)
    print("🗂️  FLUX MODELS CACHE STATUS")
    print("=" * 60)
    
    stats = get_cache_stats()
    
    print(f"📊 Cache Summary:")
    print(f"   Total models: {stats['total_models']}")
    print(f"   Cached models: {stats['cached_models']}")
    print(f"   Cache percentage: {stats['cache_percentage']:.1f}%")
    print(f"   Total cache size: {stats['total_size_formatted']}")
    print()
    
    print("📋 Individual Models:")
    for model_name, info in stats["models"].items():
        status = "✅ Cached" if info["cached"] else "❌ Not cached"
        size_info = f"({info['size_formatted']})" if info["cached"] else ""
        print(f"   {model_name}: {status} {size_info}")
    
    print("=" * 60)

def cleanup_cache(dry_run=True):
    """
    Clean up unused cache files.
    
    Args:
        dry_run (bool): If True, only show what would be deleted
        
    Returns:
        dict: Cleanup statistics
    """
    cache_dir = get_hf_cache_dir()
    
    if not cache_dir.exists():
        return {"deleted_files": 0, "freed_space": 0}
    
    # For now, just return stats without actual cleanup
    # This would need more sophisticated logic to safely clean
    print("🧹 Cache cleanup functionality would go here")
    print("   (Requires careful implementation to avoid breaking active models)")
    
    return {"deleted_files": 0, "freed_space": 0}

if __name__ == "__main__":
    # Show cache status when run directly
    print_cache_status()