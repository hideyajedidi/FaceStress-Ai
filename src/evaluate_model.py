import torch
import torch.nn as nn
import json
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 📌 CONFIG
# =====================================================================
class Config:
    DATA_DIR = Path("../data/final_dataset")
    MODELS_DIR = Path("../models")  
    RESULTS_DIR = Path("../results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    CLASSES = ["normal", "stress", "fatigue"]   
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# =====================================================================
# 📌 Dataset Loader
# =====================================================================
class FaceStressDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.data = []
        root_dir = Path(root_dir) / split

        for idx, cls in enumerate(config.CLASSES):
            class_path = root_dir / cls
            if not class_path.exists():
                continue
            for img in class_path.glob("*.jpg"):
                self.data.append((img, idx))

        print(f"📁 Chargé : {split} — {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# =====================================================================
# 📌 Charger le modèle MobileNetV2
# =====================================================================
def load_trained_model(model_path):
    print(f"\n📥 Chargement du modèle : {model_path.name}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.CLASSES))

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    state_dict = checkpoint["model_state_dict"]

    # Ignorer la dernière couche si elle ne correspond pas
    for k in list(state_dict.keys()):
        if "classifier" in k:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(config.DEVICE)
    model.eval()

    print("✔ Modèle MobileNetV2 chargé avec succès.")
    return model

# =====================================================================
# 📌 Évaluation
# =====================================================================
def evaluate_model(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

# =====================================================================
# 📌 Matrice de confusion
# =====================================================================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        cmap="Blues"
    )
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.title("Matrice de confusion")

    out_path = config.RESULTS_DIR / "confusion_matrix_final.png"
    plt.savefig(out_path)
    plt.close()
    print(f"📁 Matrice de confusion enregistrée : {out_path}")

# =====================================================================
# 📌 Exemples de prédictions
# =====================================================================
def visualize_predictions(model, loader):
    model.eval()
    imgs_all, labels_all = next(iter(loader))
    imgs_all = imgs_all.to(config.DEVICE)

    with torch.no_grad():
        outputs = model(imgs_all)
        _, preds = torch.max(outputs, 1)

    num_imgs = min(9, len(imgs_all))
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))

    for i in range(num_imgs):
        img = imgs_all[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        ax[i//3][i%3].imshow(img)
        ax[i//3][i%3].set_title(
            f"GT: {config.CLASSES[labels_all[i]]}\nPred: {config.CLASSES[preds[i]]}"
        )
        ax[i//3][i%3].axis("off")

    for j in range(num_imgs, 9):
        ax[j//3][j%3].axis("off")

    out_path = config.RESULTS_DIR / "prediction_examples_final.png"
    plt.savefig(out_path)
    plt.close()
    print(f"📁 Exemples de prédictions enregistrés : {out_path}")

# =====================================================================
# 📌 MAIN
# =====================================================================
def main():
    print("\n====================================================")
    print("🎯 FACESTRESS AI — ÉVALUATION AUTOMATIQUE")
    print("====================================================")

    model_files = sorted(
        config.MODELS_DIR.glob("facestress_final_*.pth"),
        key=lambda p: p.stat().st_mtime
    )

    if not model_files:
        print("❌ Aucun modèle trouvé dans :", config.MODELS_DIR)
        return

    best_model_path = model_files[-1]
    print(f"\n📌 Modèle sélectionné : {best_model_path.name}\n")

    model = load_trained_model(best_model_path)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = FaceStressDataset(config.DATA_DIR, "test", test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    if len(test_dataset) == 0:
        print("❌ Le dossier test est vide.")
        return

    y_true, y_pred = evaluate_model(model, test_loader)

    metrics = classification_report(
        y_true, y_pred,
        target_names=config.CLASSES,
        output_dict=True,
        zero_division=0
    )

    json_path = config.RESULTS_DIR / "test_metrics_final.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📁 Métriques enregistrées : {json_path}")

    plot_confusion_matrix(y_true, y_pred)
    visualize_predictions(model, test_loader)

    print("\n====================================================")
    print("✅ ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("====================================================")
    print("📌 Résultats disponibles dans :", config.RESULTS_DIR)

if __name__ == "__main__":
    main()
