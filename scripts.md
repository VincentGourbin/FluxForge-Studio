# 📄 Scripts Documentation - FluxForge Studio

Comprehensive documentation for all utility scripts in FluxForge Studio.

## 🎯 Classification des Scripts

### ✅ **Scripts Utiles et Documentés**

#### 🗂️ **Scripts de Gestion de Cache**

##### 1. `cache_summary.py` - Rapport Complet du Cache
**Utilité**: ⭐⭐⭐⭐⭐ **ESSENTIEL**
```bash
python cache_summary.py
```
**Fonction**:
- Affichage détaillé du statut de cache de tous les modèles FLUX
- Analyse des opportunités d'optimisation
- Recommandations automatiques pour améliorer l'efficacité
- Calcul des économies d'espace possibles

**Sortie Type**:
```
🗂️ MFLUX-GRADIO CACHE SUMMARY
📊 CURRENT STATUS:
   Models cached: 7/9 (77.8%)
   Cache size: 142.3 GB
⚠️  Some models missing:
   ❌ black-forest-labs/FLUX.1-Redux-dev
💡 Run: python optimize_cache.py --predownload
```

##### 2. `check_cache.py` - Vérification Rapide du Cache
**Utilité**: ⭐⭐⭐⭐ **TRÈS UTILE**
```bash
python check_cache.py
```
**Fonction**:
- Vérification rapide de l'état du cache
- Identification immédiate des modèles manquants
- Parfait pour validation avant utilisation hors ligne

**Sortie Type**:
```
🔍 Quick Cache Check for MFLUX-Gradio
🎉 ALL MODELS CACHED! Ready to use offline.
📊 Total: 9/9 models
💾 Size: 142.3 GB
```

##### 3. `optimize_cache.py` - Optimisation Active du Cache
**Utilité**: ⭐⭐⭐⭐⭐ **ESSENTIEL**
```bash
# Voir le statut
python optimize_cache.py --status

# Télécharger les modèles manquants
python optimize_cache.py --predownload

# Télécharger un modèle spécifique
python optimize_cache.py --model "black-forest-labs/FLUX.1-Redux-dev"
```
**Fonction**:
- Pre-téléchargement automatique des modèles manquants
- Vérification de l'état du cache
- Détection automatique des modèles obsolètes
- Optimisation proactive de l'espace disque

##### 4. `cleanup_obsolete_models.py` - Nettoyage des Modèles Obsolètes
**Utilité**: ⭐⭐⭐⭐ **TRÈS UTILE**
```bash
python cleanup_obsolete_models.py
```
**Fonction**:
- Suppression sécurisée des modèles standalone remplacés par LoRA
- Économie d'espace disque significative (40+ GB)
- Confirmation interactive avant suppression
- Calculation précise de l'espace libéré

**Exemple de Gain**:
```
📦 Found: models--black-forest-labs--FLUX.1-Canny-dev
   Size: 40.2 GB
   Remove this model? (y/N): y
   ✅ Removed successfully
🎉 Cleanup completed!
💾 Total space freed: 40.2 GB
```

##### 5. `show_cleanup_savings.py` - Analyse des Économies Possibles
**Utilité**: ⭐⭐⭐ **UTILE**
```bash
python show_cleanup_savings.py
```
**Fonction**:
- Analyse détaillée des économies d'espace possibles
- Comparaison des tailles entre modèles standalone et LoRA
- Simulation sans suppression réelle



#### 🔧 **Scripts de Maintenance**


### 🧪 **Scripts de Test et Développement**

##### 6. `test_flux_redux_integration.py` - Test d'Intégration Redux
**Utilité**: ⭐⭐ **DÉVELOPPEMENT**
```bash
python test_flux_redux_integration.py
```
**Fonction**:
- Validation de l'intégration FLUX Redux
- Tests automatisés des imports
- Vérification de la cohérence des fonctions
- Rapports de compatibilité

##### 7. `test_model_cache_update.py` - Test Cache Modèles
**Utilité**: ⭐⭐ **DÉVELOPPEMENT**
```bash
python test_model_cache_update.py
```
**Fonction**:
- Validation de la liste des modèles en cache
- Tests de complétude de la documentation
- Vérification de la cohérence architecture

### ⚠️ **Scripts à Nettoyer/Archiver**


## 📊 **Recommandations d'Utilisation**

### 🚀 **Workflow Optimal**

#### **1. Installation/Premier Lancement**
```bash
# 1. Vérifier l'état initial
python check_cache.py

# 2. Optimiser le cache
python optimize_cache.py --predownload

# 3. Nettoyer les modèles obsolètes
python cleanup_obsolete_models.py

# 4. Vérifier le résultat final
python cache_summary.py
```

#### **2. Maintenance Régulière**
```bash
# Vérification mensuelle
python cache_summary.py

# Si nécessaire, optimisation
python optimize_cache.py --predownload
```

#### **3. Formation LoRA**
```bash
# Interface graphique (recommandé)
python main.py  # Onglet "Train"

# Ligne de commande (avancé)
python train.py --train-config config.json
```

### 📁 **Organisation Recommandée**

#### **Scripts à Conserver** (6 scripts)
- ✅ `cache_summary.py`
- ✅ `check_cache.py` 
- ✅ `optimize_cache.py`
- ✅ `cleanup_obsolete_models.py`
- ✅ `show_cleanup_savings.py`
- ✅ `main.py` (point d'entrée principal)

#### **Scripts de Développement** (à déplacer vers `dev/`)
- 🔧 `test_flux_redux_integration.py`
- 🔧 `test_model_cache_update.py`

#### **Scripts à Supprimer** (doublons/obsolètes)
- ❌ `check_cache_status.py` (doublon de check_cache.py)
- ❌ `setup_optimized_cache.py` (obsolète)

## 🎯 **Scripts Prioritaires pour Utilisateurs**

### **🥇 Essentiels (utilisation quotidienne)**
1. `main.py` - Application principale (avec gestion cache HuggingFace intégrée)
2. `check_cache.py` - Vérification rapide
3. `optimize_cache.py` - Gestion des modèles

### **🥈 Utiles (utilisation occasionnelle)**
4. `cache_summary.py` - Analyse détaillée
5. `cleanup_obsolete_models.py` - Optimisation espace
6. `show_cleanup_savings.py` - Analyse économies

---

## 📋 **Actions Recommandées**

### ✅ **À Faire Immédiatement**
1. **Supprimer doublons**: `check_cache_status.py`, `setup_optimized_cache.py`
2. **Créer dossier dev/**: Déplacer scripts de test
3. **Mettre à jour noms**: Remplacer "MFLUX-Gradio" par "FluxForge Studio"
4. **Tester workflow**: Valider le processus d'optimisation cache

### 📚 **Documentation Additionnelle**
- Guide d'installation avec workflow cache
- Tutoriel formation LoRA avec train.py
- FAQ pour résolution problèmes cache
- Scripts d'automatisation pour CI/CD

Cette documentation constitue un guide complet pour l'utilisation efficace de tous les scripts de FluxForge Studio ! 🎉