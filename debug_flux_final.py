#!/usr/bin/env python3
"""
Script de debug final FLUX avec mesure mémoire et quantification optimisée

Tests:
1. Génération standard avec différents dtypes (float32, float16 sur MPS)
2. Génération quantifiée avec type compatible selon device
3. Mesure consommation mémoire réelle
4. Nettoyage complet entre tests
"""

import os
import sys
import warnings
import torch
import datetime
import gc
import psutil
import time
from pathlib import Path

# Configuration environnement
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Supprimer warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers.*")

def get_memory_usage(device):
    """Mesure la consommation mémoire selon le device"""
    try:
        if device == 'mps':
            # Mémoire GPU MPS
            gpu_memory = torch.mps.current_allocated_memory() / 1024**3  # GB
            return f"GPU: {gpu_memory:.2f} GB"
        elif device == 'cuda':
            # Mémoire GPU CUDA
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            return f"GPU: {gpu_memory:.2f} GB"
        else:
            # Mémoire RAM CPU
            ram_memory = psutil.Process().memory_info().rss / 1024**3  # GB
            return f"RAM: {ram_memory:.2f} GB"
    except Exception as e:
        return f"Erreur mesure: {e}"

def deep_cleanup(device):
    """Nettoyage mémoire complet selon device"""
    try:
        # Force garbage collection
        gc.collect()
        
        if device == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Double garbage collection après vidage cache
        gc.collect()
        
        print("   🧹 Nettoyage mémoire effectué")
        
    except Exception as e:
        print(f"   ⚠️  Nettoyage partiel: {e}")

def check_environment():
    """Vérifier l'environnement et les dépendances"""
    print("=" * 70)
    print("🔍 DIAGNOSTIC ENVIRONNEMENT COMPLET")
    print("=" * 70)
    
    # Device detection
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Device sélectionné: {device}")
    print(f"📊 Mémoire initiale: {get_memory_usage(device)}")
    
    # Test imports
    try:
        from diffusers import FluxPipeline
        print("✅ diffusers.FluxPipeline disponible")
    except ImportError as e:
        print(f"❌ diffusers manquant: {e}")
        return False, None
    
    # Test quantification
    quantization_available = False
    try:
        from optimum.quanto import quantize, freeze, qfloat8, qint8, qint4
        # Test import int2 avec gestion d'erreur
        try:
            from optimum.quanto import qint2
            int2_available = True
            print("✅ optimum.quanto disponible (avec qint2)")
        except ImportError:
            int2_available = False
            print("✅ optimum.quanto disponible (sans qint2)")
        
        quantization_available = True
        
        # Configuration selon device
        if device == 'mps':
            print("🔧 Config MPS: qint8, qint4", "+ qint2" if int2_available else "")
        elif device == 'cuda':
            print("🔧 Config CUDA: qfloat8, qint8, qint4", "+ qint2" if int2_available else "")
        else:
            print("🔧 Config CPU: qint8, qint4", "+ qint2" if int2_available else "")
            
    except ImportError as e:
        print(f"⚠️  optimum.quanto manquant: {e}")
        int2_available = False
    
    print(f"🔧 Quantification: {'✅ Disponible' if quantization_available else '❌ Non disponible'}")
    print()
    
    return True, {
        "device": device,
        "quantization_available": quantization_available,
        "int2_available": int2_available,
        "FluxPipeline": FluxPipeline
    }

def test_dtype_comparison(env_info):
    """Test différents dtypes sur MPS pour trouver le meilleur"""
    device = env_info["device"]
    FluxPipeline = env_info["FluxPipeline"]
    
    if device != 'mps':
        print("⏭️  Test dtype réservé à MPS, device actuel:", device)
        return None
    
    print("=" * 70)
    print("🧪 TEST DTYPE COMPARISON SUR MPS")
    print("=" * 70)
    
    results = {}
    dtypes_to_test = [
        (torch.float32, "float32", "sûr, fix images noires"),
        (torch.float16, "float16", "half precision standard"),
        (torch.bfloat16, "bfloat16", "brain float, utilisé avant dans le projet")
    ]
    
    for dtype, name, description in dtypes_to_test:
        print(f"\n🔧 Test avec {name} ({description})...")
        
        try:
            # Mesure mémoire avant
            memory_before = get_memory_usage(device)
            print(f"   📊 Mémoire avant: {memory_before}")
            
            # Charger pipeline
            print("   📦 Chargement pipeline...")
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=dtype,
                use_safetensors=True
            )
            pipeline = pipeline.to(device)
            pipeline.enable_attention_slicing()
            
            # Mesure mémoire après chargement
            memory_loaded = get_memory_usage(device)
            print(f"   📊 Mémoire chargé: {memory_loaded}")
            
            # Génération test
            generator = torch.Generator(device=device).manual_seed(42)
            
            with torch.inference_mode():
                result = pipeline(
                    prompt="a red apple on a white table",
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    generator=generator,
                    guidance_scale=3.5
                )
                image = result.images[0]
            
            # Analyse qualité
            import numpy as np
            img_array = np.array(image)
            brightness = np.mean(img_array)
            
            # Sauvegarde
            timestamp = datetime.datetime.now().strftime('%H%M%S')
            filename = f"test_{name}_{timestamp}_bright_{brightness:.0f}.png"
            image.save(filename)
            
            # Résultats
            status = "✅ OK" if brightness > 10 else "❌ NOIR"
            results[name] = {
                "brightness": brightness,
                "status": status,
                "filename": filename,
                "memory_before": memory_before,
                "memory_loaded": memory_loaded
            }
            
            print(f"   🎨 Résultat: {status} (luminosité: {brightness:.1f})")
            print(f"   💾 Sauvé: {filename}")
            
            # Nettoyage
            del pipeline, result, image
            deep_cleanup(device)
            
        except Exception as e:
            print(f"   ❌ Erreur {name}: {e}")
            results[name] = {"status": "❌ ERREUR", "error": str(e)}
            deep_cleanup(device)
    
    # Comparaison finale
    print(f"\n📊 COMPARAISON DTYPES:")
    for dtype_name, result in results.items():
        if "brightness" in result:
            print(f"   {dtype_name}: {result['status']} - {result['brightness']:.1f}")
        else:
            print(f"   {dtype_name}: {result['status']}")
    
    return results

def test_standard_vs_quantized(env_info, best_dtype=torch.float32):
    """Test génération standard vs quantifiée avec mesure mémoire"""
    device = env_info["device"]
    FluxPipeline = env_info["FluxPipeline"]
    
    print("=" * 70)
    print("🔬 TEST STANDARD VS QUANTIFICATIONS MULTIPLES")
    print("=" * 70)
    
    results = {}
    
    # Configuration selon device - maintenant avec plusieurs options de quantification
    if device == 'mps':
        dtype = best_dtype
        # MPS supporte qint8, qint4, et potentiellement qint2
        quant_configs = [
            ("qint8", "8-bit integer"),
            ("qint4", "4-bit integer (ultra compression)")
        ]
        if env_info.get("int2_available", False):
            quant_configs.append(("qint2", "2-bit integer (extreme compression)"))
    elif device == 'cuda':
        dtype = torch.bfloat16
        # CUDA supporte tous les formats
        quant_configs = [
            ("qfloat8", "8-bit float (performance optimale)"),
            ("qint8", "8-bit integer"),
            ("qint4", "4-bit integer (ultra compression)")
        ]
        if env_info.get("int2_available", False):
            quant_configs.append(("qint2", "2-bit integer (extreme compression)"))
    else:
        dtype = torch.float32
        # CPU: formats conservateurs
        quant_configs = [
            ("qint8", "8-bit integer"),
            ("qint4", "4-bit integer")
        ]
        if env_info.get("int2_available", False):
            quant_configs.append(("qint2", "2-bit integer (extreme compression)"))
    
    print(f"🎯 Device: {device}, dtype={dtype}")
    print(f"🔬 Tests de quantification: {len(quant_configs)} types")
    for name, desc in quant_configs:
        print(f"   - {name}: {desc}")
    
    # TEST 1: Standard
    print(f"\n1️⃣ GÉNÉRATION STANDARD")
    try:
        time_total_start = time.time()
        
        memory_start = get_memory_usage(device)
        print(f"   📊 Mémoire départ: {memory_start}")
        
        # Charger pipeline standard
        time_loading_start = time.time()
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipeline = pipeline.to(device)
        pipeline.enable_attention_slicing()
        time_loading_end = time.time()
        
        memory_loaded = get_memory_usage(device)
        print(f"   📊 Mémoire après chargement: {memory_loaded}")
        print(f"   ⏱️  Temps chargement: {time_loading_end - time_loading_start:.2f}s")
        
        # Génération
        generator = torch.Generator(device=device).manual_seed(42)
        time_generation_start = time.time()
        with torch.inference_mode():
            result = pipeline(
                prompt="a red apple on a white table",
                width=512, height=512,
                num_inference_steps=4,
                generator=generator,
                guidance_scale=3.5
            )
            image = result.images[0]
        time_generation_end = time.time()
        
        memory_peak = get_memory_usage(device)
        print(f"   📊 Mémoire pic génération: {memory_peak}")
        print(f"   ⏱️  Temps génération: {time_generation_end - time_generation_start:.2f}s")
        
        # Analyse
        import numpy as np
        brightness = np.mean(np.array(image))
        
        time_total_end = time.time()
        
        timestamp = datetime.datetime.now().strftime('%H%M%S')
        filename = f"standard_{timestamp}_bright_{brightness:.0f}.png"
        image.save(filename)
        
        results['standard'] = {
            "brightness": brightness,
            "filename": filename,
            "memory_start": memory_start,
            "memory_loaded": memory_loaded,
            "memory_peak": memory_peak,
            "time_loading": time_loading_end - time_loading_start,
            "time_generation": time_generation_end - time_generation_start,
            "time_total": time_total_end - time_total_start,
            "status": "✅ OK" if brightness > 10 else "❌ NOIR"
        }
        
        print(f"   🎨 Standard: {results['standard']['status']} (luminosité: {brightness:.1f})")
        print(f"   ⏱️  Temps total: {results['standard']['time_total']:.2f}s")
        print(f"   💾 Sauvé: {filename}")
        
        # Nettoyage complet
        del pipeline, result, image
        deep_cleanup(device)
        
    except Exception as e:
        print(f"   ❌ Erreur standard: {e}")
        results['standard'] = {"status": "❌ ERREUR", "error": str(e)}
        deep_cleanup(device)
    
    # TEST 2+: Quantifications multiples (si disponible)
    if env_info["quantization_available"]:
        from optimum.quanto import quantize, freeze, qfloat8, qint8, qint4
        # Import conditionnel pour qint2
        qint2 = None
        if env_info.get("int2_available", False):
            try:
                from optimum.quanto import qint2
            except ImportError:
                qint2 = None
        
        for i, (quant_name, quant_desc) in enumerate(quant_configs, 2):
            print(f"\n{i}️⃣ GÉNÉRATION QUANTIFIÉE ({quant_name})")
            print(f"   📄 Description: {quant_desc}")
            
            try:
                # Sélection du type de quantification
                if quant_name == "qfloat8":
                    quant_type = qfloat8
                elif quant_name == "qint8":
                    quant_type = qint8
                elif quant_name == "qint4":
                    quant_type = qint4
                elif quant_name == "qint2":
                    if qint2 is None:
                        print(f"   ❌ qint2 non disponible")
                        continue
                    quant_type = qint2
                else:
                    print(f"   ❌ Type quantification inconnu: {quant_name}")
                    continue
                
                time_total_start = time.time()
                
                memory_start = get_memory_usage(device)
                print(f"   📊 Mémoire départ: {memory_start}")
                
                # Charger pipeline
                time_loading_start = time.time()
                pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                time_loading_end = time.time()
                
                memory_before_quant = get_memory_usage(device)
                print(f"   📊 Mémoire avant quantification: {memory_before_quant}")
                print(f"   ⏱️  Temps chargement: {time_loading_end - time_loading_start:.2f}s")
                
                # Quantification avec mesure de temps
                print(f"   🔧 Application quantification {quant_name}...")
                time_quantization_start = time.time()
                if hasattr(pipeline, 'transformer'):
                    quantize(pipeline.transformer, weights=quant_type)
                    freeze(pipeline.transformer)
                if hasattr(pipeline, 'text_encoder_2'):
                    quantize(pipeline.text_encoder_2, weights=quant_type)
                    freeze(pipeline.text_encoder_2)
                
                pipeline = pipeline.to(device)
                pipeline.enable_attention_slicing()
                time_quantization_end = time.time()
                
                memory_loaded = get_memory_usage(device)
                print(f"   📊 Mémoire après quantification: {memory_loaded}")
                print(f"   ⏱️  Temps quantification: {time_quantization_end - time_quantization_start:.2f}s")
                
                # Génération avec mesure de temps
                generator = torch.Generator(device=device).manual_seed(42)
                time_generation_start = time.time()
                with torch.inference_mode():
                    result = pipeline(
                        prompt="a red apple on a white table",
                        width=512, height=512,
                        num_inference_steps=4,
                        generator=generator,
                        guidance_scale=3.5
                    )
                    image = result.images[0]
                time_generation_end = time.time()
                
                memory_peak = get_memory_usage(device)
                print(f"   📊 Mémoire pic génération: {memory_peak}")
                print(f"   ⏱️  Temps génération: {time_generation_end - time_generation_start:.2f}s")
                
                # Analyse
                import numpy as np
                brightness = np.mean(np.array(image))
                
                time_total_end = time.time()
                
                timestamp = datetime.datetime.now().strftime('%H%M%S')
                filename = f"quantized_{quant_name}_{timestamp}_bright_{brightness:.0f}.png"
                image.save(filename)
                
                results[f'quantized_{quant_name}'] = {
                    "brightness": brightness,
                    "filename": filename,
                    "memory_start": memory_start,
                    "memory_before_quant": memory_before_quant,
                    "memory_loaded": memory_loaded,
                    "memory_peak": memory_peak,
                    "time_loading": time_loading_end - time_loading_start,
                    "time_quantization": time_quantization_end - time_quantization_start,
                    "time_generation": time_generation_end - time_generation_start,
                    "time_total": time_total_end - time_total_start,
                    "status": "✅ OK" if brightness > 10 else "❌ NOIR",
                    "quant_type": quant_name,
                    "quant_desc": quant_desc
                }
                
                print(f"   🎨 Quantifié {quant_name}: {results[f'quantized_{quant_name}']['status']} (luminosité: {brightness:.1f})")
                print(f"   ⏱️  Temps total: {results[f'quantized_{quant_name}']['time_total']:.2f}s")
                print(f"      └── Chargement: {results[f'quantized_{quant_name}']['time_loading']:.2f}s")
                print(f"      └── Quantification: {results[f'quantized_{quant_name}']['time_quantization']:.2f}s")
                print(f"      └── Génération: {results[f'quantized_{quant_name}']['time_generation']:.2f}s")
                print(f"   💾 Sauvé: {filename}")
                
                # Nettoyage complet
                del pipeline, result, image
                deep_cleanup(device)
                
            except Exception as e:
                print(f"   ❌ Erreur quantification {quant_name}: {e}")
                results[f'quantized_{quant_name}'] = {
                    "status": "❌ ERREUR", 
                    "error": str(e),
                    "quant_type": quant_name
                }
                deep_cleanup(device)
    else:
        print(f"\n2️⃣ QUANTIFICATION IGNORÉE (optimum.quanto non disponible)")
    
    return results

def analyze_results(dtype_results, memory_results):
    """Analyse finale des résultats"""
    print("=" * 70)
    print("📊 ANALYSE FINALE DES RÉSULTATS")
    print("=" * 70)
    
    # Dtype comparison
    if dtype_results:
        print("\n🎯 MEILLEUR DTYPE SUR MPS:")
        for dtype_name, result in dtype_results.items():
            if "brightness" in result:
                print(f"   {dtype_name}: {result['status']} (luminosité: {result['brightness']:.1f})")
    
    # Memory comparison
    if memory_results:
        print("\n💾 CONSOMMATION MÉMOIRE:")
        
        std_memory = None
        if 'standard' in memory_results and "memory_peak" in memory_results['standard']:
            std = memory_results['standard']
            print(f"   Standard: {std['memory_peak']}")
            try:
                std_memory = float(std['memory_peak'].split()[1])
            except:
                pass
        
        # Afficher toutes les quantifications testées
        quantized_results = [(k, v) for k, v in memory_results.items() if k.startswith('quantized_')]
        
        if quantized_results:
            print("\n   🔬 Quantifications testées:")
            best_saving = 0
            best_quant = None
            
            for quant_key, quant_data in quantized_results:
                if "memory_peak" in quant_data and quant_data['status'] == "✅ OK":
                    quant_type = quant_data['quant_type']
                    quant_desc = quant_data.get('quant_desc', '')
                    memory_peak = quant_data['memory_peak']
                    brightness = quant_data.get('brightness', 0)
                    time_total = quant_data.get('time_total', 0)
                    time_generation = quant_data.get('time_generation', 0)
                    time_quantization = quant_data.get('time_quantization', 0)
                    
                    print(f"      {quant_type}: {memory_peak} (luminosité: {brightness:.1f})")
                    print(f"         ⏱️  Temps total: {time_total:.2f}s | génération: {time_generation:.2f}s | quantification: {time_quantization:.2f}s")
                    
                    # Calcul économie mémoire
                    if std_memory:
                        try:
                            quant_mem = float(memory_peak.split()[1])
                            saving = ((std_memory - quant_mem) / std_memory) * 100
                            print(f"         💾 Économie mémoire: {saving:.1f}%")
                            
                            if saving > best_saving:
                                best_saving = saving
                                best_quant = quant_type
                        except:
                            print("         💾 Économie mémoire: calcul impossible")
                    
                    # Comparaison temps avec standard
                    if 'standard' in memory_results and 'time_generation' in memory_results['standard']:
                        std_time = memory_results['standard']['time_generation']
                        time_ratio = (time_generation / std_time) * 100
                        if time_ratio < 100:
                            print(f"         🚀 {100-time_ratio:.1f}% plus rapide que standard")
                        elif time_ratio > 100:
                            print(f"         🐌 {time_ratio-100:.1f}% plus lent que standard")
                        else:
                            print(f"         ⚖️  Même vitesse que standard")
                            
                elif quant_key in memory_results:
                    quant_type = memory_results[quant_key].get('quant_type', quant_key)
                    status = memory_results[quant_key].get('status', 'Unknown')
                    print(f"      {quant_type}: {status}")
            
            if best_quant:
                print(f"\n   🏆 Meilleure quantification: {best_quant} ({best_saving:.1f}% économie mémoire)")
    
    # Résumé temps de génération
    if memory_results:
        print("\n⏱️  TEMPS DE GÉNÉRATION:")
        if 'standard' in memory_results and 'time_generation' in memory_results['standard']:
            std_time = memory_results['standard']['time_generation']
            print(f"   Standard: {std_time:.2f}s")
            
            # Trouver le plus rapide
            fastest_quant = None
            fastest_time = std_time
            for quant_key, quant_data in quantized_results:
                if quant_data.get('status') == "✅ OK" and 'time_generation' in quant_data:
                    if quant_data['time_generation'] < fastest_time:
                        fastest_time = quant_data['time_generation']
                        fastest_quant = quant_data['quant_type']
            
            if fastest_quant:
                speedup = ((std_time - fastest_time) / std_time) * 100
                print(f"   🏃 Plus rapide: {fastest_quant} ({fastest_time:.2f}s, +{speedup:.1f}% gain)")
    
    # Recommandations
    print("\n🎯 RECOMMANDATIONS:")
    
    if dtype_results:
        working_dtypes = [name for name, result in dtype_results.items() 
                         if result.get('brightness', 0) > 10]
        if working_dtypes:
            print(f"   ✅ Dtypes fonctionnels: {', '.join(working_dtypes)}")
        else:
            print("   ❌ Aucun dtype ne fonctionne correctement")
    
    if memory_results:
        # Compter les quantifications qui fonctionnent
        working_quants = [k for k, v in memory_results.items() 
                         if k.startswith('quantized_') and v.get('status') == "✅ OK"]
        
        if working_quants:
            print(f"   ✅ {len(working_quants)} quantification(s) fonctionnelle(s)")
            if best_quant:
                print(f"   🏆 Recommandation: {best_quant} (meilleure économie mémoire)")
        elif 'standard' in memory_results and memory_results['standard']['status'] == "✅ OK":
            print("   ⚠️  Utiliser mode standard uniquement (quantifications problématiques)")
        else:
            print("   ❌ Problèmes détectés, investigation nécessaire")

def main():
    """Fonction principale du test final optimisé"""
    print("🚀 DEBUG FLUX FINAL - OPTIMISÉ AVEC MESURE MÉMOIRE")
    print(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Tests: dtype comparison + quantification + mesure mémoire")
    print()
    
    # Diagnostic environnement
    env_ok, env_info = check_environment()
    if not env_ok:
        print("❌ Environnement incompatible")
        return
    
    # Test dtype sur MPS
    dtype_results = test_dtype_comparison(env_info)
    
    # Détermine le meilleur dtype
    best_dtype = torch.float32  # Défaut sûr
    if dtype_results:
        for dtype_name, result in dtype_results.items():
            if result.get('brightness', 0) > 10:
                if dtype_name == 'float16':
                    best_dtype = torch.float16  # Préfère float16 si ça marche
                    break
    
    # Test standard vs quantifié avec mesure mémoire
    memory_results = test_standard_vs_quantized(env_info, best_dtype)
    
    # Analyse finale
    analyze_results(dtype_results, memory_results)
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés - Vérifiez les images générées et mesures mémoire")
    print("=" * 70)

if __name__ == "__main__":
    main()