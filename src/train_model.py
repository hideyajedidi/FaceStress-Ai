"""
FaceStress AI - Fine-tuning du modèle
Semaine 3 : Entraînement avec Transfer Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime



class Config:
    # Chemins
    BASE_DIR = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Paramètres d'entraînement
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Classes
    # Doivent correspondre à preprocessing.py
    CLASSES = ['stress', 'tristesse', 'joie', 'colère', 'surprise', 'neutre']
    NUM_CLASSES = len(CLASSES) # Devrait être 6
    
    # Device (GPU si disponible)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modèle
    MODEL_NAME = 'mobilenet_v2'  # ou 'resnet18'
    
    # Early stopping
    PATIENCE = 5
    
config = Config()

# Créer les dossiers
(config.MODELS_DIR / "finetuned").mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR / "visualizations").mkdir(parents=True, exist_ok=True)

print(f"🖥️  Device : {config.DEVICE}")
print(f"📦 Batch size : {config.BATCH_SIZE}")
print(f"🔄 Epochs : {config.NUM_EPOCHS}")


# ============================================
# 2. DATASET CUSTOM
# ============================================

class FaceStressDataset(Dataset):
    """Dataset custom pour FaceStress AI"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = config.CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Charger tous les chemins d'images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"  {split:5s} : {len(self.samples):6d} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================
# 3. TRANSFORMATIONS
# ============================================

def get_transforms():
    """Retourne les transformations pour train et val/test"""
    
    # Normalisation ImageNet (pour modèles pré-entraînés)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Augmentation pour train
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    # Sans augmentation pour val/test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, eval_transform


# ============================================
# 4. MODÈLE
# ============================================

def create_model(model_name='mobilenet_v2', num_classes=3):
    """Crée le modèle avec transfer learning"""
    
    print(f"\n📥 Chargement du modèle : {model_name}")
    
    if model_name == 'mobilenet_v2':
        # Charger MobileNetV2 pré-entraîné
        model = models.mobilenet_v2(pretrained=True)
        
        # Geler les couches pré-entraînées
        for param in model.parameters():
            param.requires_grad = False
        
        # Remplacer la dernière couche (classifier)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name == 'resnet18':
        # Charger ResNet18 pré-entraîné
        model = models.resnet18(pretrained=True)
        
        # Geler les couches
        for param in model.parameters():
            param.requires_grad = False
        
        # Remplacer fc (fully connected)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    else:
        raise ValueError(f"Modèle {model_name} non supporté")
    
    print(f"✅ Modèle créé : {model_name}")
    print(f"   Nombre de classes : {num_classes}")
    
    return model


# ============================================
# 5. ENTRAÎNEMENT
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Valide le modèle"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Boucle d'entraînement complète"""
    
    print(f"\n🚀 Début de l'entraînement...")
    print("="*70)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = config.MODELS_DIR / "finetuned" / f"facestress_best_{timestamp}.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': config.CLASSES
            }, best_model_path)
            
            print(f"✅ Meilleur modèle sauvegardé : {best_model_path.name}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️  Early stopping : pas d'amélioration depuis {config.PATIENCE} epochs")
            break
    
    print("\n" + "="*70)
    print(f"✅ Entraînement terminé !")
    print(f"🏆 Meilleure précision validation : {best_val_acc:.2f}%")
    print(f"💾 Modèle sauvegardé : {best_model_path}")
    
    return history, best_model_path


# ============================================
# 6. VISUALISATION
# ============================================

def plot_training_history(history):
    """Visualise l'historique d'entraînement"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Loss Evolution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_title('Accuracy Evolution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = config.RESULTS_DIR / 'visualizations' / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé : {output_path}")
    plt.close()


# ============================================
# 7. EXÉCUTION PRINCIPALE
# ============================================

def main():
    print("\n" + "="*70)
    print("🎯 FACESTRESS AI - FINE-TUNING")
    print("="*70)
    
    # 1. Transformations
    train_transform, eval_transform = get_transforms()
    
    # 2. Datasets
    print("\n📁 Chargement des datasets...")
    train_dataset = FaceStressDataset(config.DATA_DIR, 'train', train_transform)
    val_dataset = FaceStressDataset(config.DATA_DIR, 'val', eval_transform)
    
    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Modèle
    model = create_model(config.MODEL_NAME, config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # 5. Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 6. Entraînement
    history, best_model_path = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, config.NUM_EPOCHS, config.DEVICE
    )
    
    # 7. Sauvegarder l'historique
    history_path = config.RESULTS_DIR / 'metrics' / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"💾 Historique sauvegardé : {history_path}")
    
    # 8. Visualisation
    plot_training_history(history)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print(f"\n📁 Modèle : {best_model_path}")
    print(f"📊 Métriques : {history_path}")
    print("\n🚀 Prochaine étape : python src/evaluate.py (évaluation)")


if __name__ == "__main__":
    main()