"""
FaceStress AI - Prétraitement des données
Version corrigée pour gérer les chemins avec espaces
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ============================================
# 1. CONFIGURATION
# ============================================

class Config:
    # Chemins - utiliser Path pour gérer les espaces
    BASE_DIR = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw" / "fer2013"
    PROCESSED_DIR = DATA_DIR / "processed"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Paramètres images
    IMG_SIZE = (224, 224)
    
    # Mapping des émotions
    EMOTION_MAPPING = {
        'angry': 'colère',
        'disgust': 'stress',
        'fear': 'stress',
        'happy': 'joie',
        'sad': 'tristesse',
        'surprise': 'surprise',
        'neutral': 'neutre'
    }
    
    # Classes cibles
    # 'fatigue' est un cas spécial, on peut le déduire de 'tristesse' ou l'ajouter si on a un dataset adapté.
    # Pour l'instant, on se base sur les émotions de FER2013.
    CLASSES = ['stress', 'tristesse', 'joie', 'colère', 'surprise', 'neutre']
    
    # Augmentation
    AUGMENT_MINORITY = True

config = Config()

# Créer les dossiers
print(f"📂 Dossier de base : {config.BASE_DIR}")
print(f"📂 Dataset source : {config.RAW_DIR}")
print(f"📂 Destination : {config.PROCESSED_DIR}")

for split in ['train', 'val', 'test']:
    for class_name in config.CLASSES:
        folder = config.PROCESSED_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# Créer dossier results
(config.RESULTS_DIR / "visualizations").mkdir(parents=True, exist_ok=True)


# ============================================
# 2. FONCTIONS
# ============================================

def preprocess_image(image_path, target_size=(224, 224)):
    """Charge et prétraite une image"""
    try:
        img = cv2.imread(str(image_path))
        
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        
        return img_resized
    except Exception as e:
        print(f"Erreur lecture {image_path}: {e}")
        return None


def augment_image(image):
    """Applique des augmentations"""
    h, w = image.shape[:2]
    augmented = []
    
    # Flip
    augmented.append(cv2.flip(image, 1))
    
    # Rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    
    # Luminosité
    brightness = random.uniform(0.7, 1.3)
    augmented.append(cv2.convertScaleAbs(image, alpha=brightness, beta=0))
    
    return augmented


# ============================================
# 3. TRAITEMENT
# ============================================

def check_dataset_exists():
    """Vérifie que le dataset existe"""
    if not config.RAW_DIR.exists():
        print(f"❌ ERREUR : {config.RAW_DIR} n'existe pas")
        return False
    
    train_path = config.RAW_DIR / "train"
    test_path = config.RAW_DIR / "test"
    
    if not train_path.exists():
        print(f"❌ ERREUR : {train_path} n'existe pas")
        return False
    
    if not test_path.exists():
        print(f"❌ ERREUR : {test_path} n'existe pas")
        return False
    
    print("✅ Dataset trouvé !")
    return True


def count_raw_images():
    """Compte les images sources"""
    print("\n📊 Comptage des images sources :")
    print("="*70)
    
    counts = {}
    
    for split in ['train', 'test']:
        split_dir = config.RAW_DIR / split
        counts[split] = {}
        
        print(f"\n{split.upper()}/ :")
        
        for emotion in config.EMOTION_MAPPING.keys():
            emotion_dir = split_dir / emotion
            
            if emotion_dir.exists():
                images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
                count = len(images)
                counts[split][emotion] = count
                mapped = config.EMOTION_MAPPING[emotion]
                print(f"  {emotion:10s} -> {mapped:8s} : {count:5d} images")
            else:
                counts[split][emotion] = 0
                print(f"  {emotion:10s} -> manquant")
    
    return counts


def process_dataset():
    """Traite le dataset complet"""
    
    print("\n🚀 DÉBUT DU TRAITEMENT")
    print("="*70)
    
    # Vérifier l'existence
    if not check_dataset_exists():
        return None
    
    # Compter les sources
    raw_counts = count_raw_images()
    
    # Compteurs finaux
    class_counts = {
        'train': {c: 0 for c in config.CLASSES},
        'val': {c: 0 for c in config.CLASSES},
        'test': {c: 0 for c in config.CLASSES}
    }
    
    # Traiter chaque split
    for split in ['train', 'test']:
        split_dir = config.RAW_DIR / split
        
        print(f"\n📁 Traitement : {split.upper()}/")
        print("-"*70)
        
        # Traiter chaque émotion
        for emotion, target_class in config.EMOTION_MAPPING.items():
            emotion_dir = split_dir / emotion
            
            if not emotion_dir.exists():
                continue
            
            # Liste des images
            image_files = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
            
            if len(image_files) == 0:
                continue
            
            print(f"\n  {emotion} -> {target_class} : {len(image_files)} images")
            
            # Barre de progression
            for idx, img_path in enumerate(tqdm(image_files, desc=f"  Processing", leave=False)):
                
                # Prétraiter
                img = preprocess_image(img_path, config.IMG_SIZE)
                
                if img is None:
                    continue
                
                # Split destination (80% train, 20% val pour split 'train')
                if split == 'train':
                    dest_split = 'train' if random.random() < 0.8 else 'val'
                else:
                    dest_split = 'test'
                
                # Sauvegarder
                dest_dir = config.PROCESSED_DIR / dest_split / target_class
                dest_path = dest_dir / f"{dest_split}_{target_class}_{emotion}_{idx:05d}.jpg"
                
                cv2.imwrite(str(dest_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                class_counts[dest_split][target_class] += 1
                
                # Augmentation (seulement train, classes minoritaires)
                if config.AUGMENT_MINORITY and dest_split == 'train' and target_class in ['stress', 'fatigue']:
                    if random.random() < 0.3:  # 30% des images
                        augmented = augment_image(img)
                        for aug_idx, aug_img in enumerate(augmented[:2], 1):
                            aug_path = dest_dir / f"{dest_split}_{target_class}_{emotion}_{idx:05d}_aug{aug_idx}.jpg"
                            cv2.imwrite(str(aug_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                            class_counts[dest_split][target_class] += 1
    
    # Statistiques finales
    print("\n" + "="*70)
    print("📊 STATISTIQUES FINALES")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        total = sum(class_counts[split].values())
        print(f"\n{split.upper()} : {total} images")
        for class_name in config.CLASSES:
            count = class_counts[split][class_name]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {class_name:10s} : {count:6d} ({pct:5.1f}%)")
    
    total_all = sum(sum(class_counts[s].values()) for s in class_counts)
    print(f"\n{'TOTAL':10s} : {total_all:6d} images")
    
    return class_counts


def visualize_samples():
    """Crée une visualisation d'exemples"""
    print("\n📸 Création de visualisations...")
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for i, class_name in enumerate(config.CLASSES):
        class_dir = config.PROCESSED_DIR / 'train' / class_name
        
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob("*.jpg"))
        
        if len(images) == 0:
            continue
        
        selected = random.sample(images, min(6, len(images)))
        
        for j, img_path in enumerate(selected):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(img)
            if j == 0:
                axes[i, j].set_ylabel(
                    class_name.upper(), 
                    fontsize=14, 
                    fontweight='bold',
                    rotation=0,
                    labelpad=50,
                    va='center'
                )
            axes[i, j].axis('off')
    
    plt.suptitle('FaceStress AI - Exemples par classe', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = config.RESULTS_DIR / 'visualizations' / 'samples_processed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualisation : {output_path}")
    plt.close()


# ============================================
# 4. EXÉCUTION
# ============================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🎯 FACESTRESS AI - PRÉTRAITEMENT")
    print("="*70)
    
    # Traiter
    class_counts = process_dataset()
    
    if class_counts:
        # Visualiser
        visualize_samples()
        
        print("\n" + "="*70)
        print("✅ PRÉTRAITEMENT TERMINÉ")
        print("="*70)
        print(f"\n📁 Données : {config.PROCESSED_DIR}")
        print(f"📊 Visualisations : {config.RESULTS_DIR / 'visualizations'}")
        print("\n🚀 Prochaine étape : python src/model.py (training)")
    else:
        print("\n❌ Échec du prétraitement")