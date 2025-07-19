# LoRA Management Guide

Ce guide explique comment utiliser le nouveau système de gestion des LoRA avec interface graphique.

## 🚀 Migration depuis le fichier JSON

### Étape 1 : Exécuter le script de migration

```bash
python migrate_lora_to_db.py
```

Ce script va :
- Créer la table `lora` dans la base de données SQLite
- Migrer toutes les données de `lora_info.json` vers la base de données
- Créer une sauvegarde de votre fichier JSON original
- Vérifier que la migration s'est bien déroulée

### Étape 2 : Vérifier la migration

Le script affichera :
- Le nombre de LoRA migrés
- Le nombre de fichiers trouvés/manquants
- Le statut final de la migration

## 📋 Utilisation de l'interface LoRA Management

### Vue d'ensemble

L'onglet **LoRA Management** remplace l'édition manuelle du fichier JSON et offre une interface graphique complète pour :
- Visualiser tous vos LoRA
- Ajouter de nouveaux LoRA
- Modifier les descriptions et mots-clés
- Supprimer des LoRA

### Interface principale

#### Section "Current LoRA Models"
- **Tableau interactif** : Affiche tous vos LoRA avec ID, nom, description, mot-clé, taille et statut
- **Bouton Refresh List** : Actualise la liste des LoRA
- **Bouton Refresh File Sizes** : Met à jour les tailles de fichiers
- **Indicateur de statut** : Affiche si les fichiers existent sur le disque

## ➕ Ajouter un nouveau LoRA

### Étapes :
1. **Upload File** : Sélectionnez un fichier `.safetensors`
2. **Description** : Ajoutez une description détaillée
3. **Activation Keyword** : Ajoutez le mot-clé d'activation (optionnel)
4. **Cliquez sur "Add LoRA"**

### Que se passe-t-il ?
- Le fichier est copié dans le dossier `lora/`
- Les métadonnées sont enregistrées en base de données
- La liste des LoRA est automatiquement mise à jour
- Le LoRA devient immédiatement disponible pour la génération

## ✏️ Modifier un LoRA existant

### Étapes :
1. **Entrez l'ID du LoRA** (visible dans le tableau)
2. **Cliquez sur "Load LoRA Details"**
3. **Modifiez** la description et/ou le mot-clé d'activation
4. **Cliquez sur "Update LoRA"**

### Informations affichées :
- ID et nom du fichier
- Taille du fichier
- Statut (existe/manquant)
- Dates de création et modification

## 🗑️ Supprimer un LoRA

### Étapes :
1. **Entrez l'ID du LoRA** à supprimer
2. **Cochez optionnellement** "Also delete file from disk"
3. **Cliquez sur "Delete LoRA"**

### Options de suppression :
- **Base de données seulement** : Supprime l'entrée mais garde le fichier
- **Base + fichier** : Supprime l'entrée ET le fichier .safetensors

⚠️ **Attention** : Cette action est irréversible !

## 🔄 Synchronisation automatique

### Après chaque opération :
- Les données LoRA sont automatiquement rafraîchies
- Les modifications sont immédiatement disponibles pour la génération
- Pas besoin de redémarrer l'application

### Indicateurs visuels :
- **✅ Exists** : Le fichier .safetensors existe
- **❌ Missing** : Le fichier est manquant du disque
- **Taille du fichier** : Taille réelle du fichier sur le disque

## 🛠️ Fonctionnalités avancées

### Refresh File Sizes
- Met à jour les tailles de fichiers pour tous les LoRA
- Utile si vous avez ajouté des fichiers manuellement dans le dossier `lora/`

### Gestion des erreurs
- Messages d'erreur détaillés en cas de problème
- Validation des fichiers (seuls les .safetensors sont acceptés)
- Vérification des doublons (noms de fichiers uniques)

## 📊 Avantages du nouveau système

### Par rapport au fichier JSON :
- ✅ **Interface graphique** : Plus d'édition manuelle de JSON
- ✅ **Upload direct** : Glisser-déposer les fichiers
- ✅ **Validation** : Vérification automatique des fichiers
- ✅ **Historique** : Dates de création/modification
- ✅ **Tailles de fichiers** : Information sur l'espace disque
- ✅ **Statut en temps réel** : Vérification de l'existence des fichiers
- ✅ **Sauvegarde** : Données sécurisées en base de données

### Compatibilité :
- 🔄 **Rétrocompatible** : Fonctionne avec vos LoRA existants
- 🔄 **Migration transparente** : Transfert automatique depuis JSON
- 🔄 **Fallback** : Retour au JSON en cas de problème de base de données

## 🔧 Dépannage

### Problèmes courants :

#### "LoRA with filename already exists"
- Un LoRA avec ce nom existe déjà
- Renommez le fichier ou supprimez l'ancien

#### "File must be a .safetensors file"
- Seuls les fichiers .safetensors sont acceptés
- Vérifiez l'extension de votre fichier

#### "Description is required"
- La description est obligatoire
- Ajoutez une description avant de sauvegarder

#### Fichier manquant (❌ Missing)
- Le fichier .safetensors n'existe pas dans le dossier `lora/`
- Uploadez à nouveau le fichier ou utilisez "Refresh File Sizes"

### Récupération d'urgence :
Si vous rencontrez des problèmes avec la base de données, le système utilise automatiquement le fichier JSON de sauvegarde créé lors de la migration.

## 📈 Utilisation recommandée

1. **Migrez d'abord** vos LoRA existants avec le script
2. **Vérifiez** que tous vos LoRA sont correctement migrés
3. **Utilisez exclusivement** l'interface graphique pour les modifications
4. **Gardez la sauvegarde JSON** au cas où
5. **Utilisez "Refresh File Sizes"** périodiquement pour maintenir la cohérence

Le système de gestion des LoRA est maintenant prêt à l'emploi ! 🎉