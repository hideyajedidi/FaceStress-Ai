"""
FaceStress AI - Interface Gradio
Version 2.1 : Améliorations et corrections
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# 1. CONFIGURATION
# ============================================

class Config:
    """Configuration centralisée de l'application"""
    BASE_DIR = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
    MODELS_DIR = BASE_DIR / "models" / "finetuned"
    RESULTS_DIR = BASE_DIR / "results"
    
    CLASSES = ['fatigue', 'normal', 'stress']
    NUM_CLASSES = len(CLASSES)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CLASS_INFO = {
        'fatigue': {'emoji': '😴', 'label': 'Fatigue', 'color': '#FFD700'},
        'normal': {'emoji': '😊', 'label': 'Normal', 'color': '#4CAF50'},
        'stress': {'emoji': '😰', 'label': 'Stress', 'color': '#FF5252'}
    }

    CLASS_ADVICE = {
        'fatigue': [
            "💤 Prenez une pause de 10-15 minutes",
            "☕ Hydratez-vous avec de l'eau ou une boisson chaude",
            "🚶 Faites une courte marche pour vous revitaliser",
            "🛌 Envisagez une sieste de 20 minutes si possible"
        ],
        'normal': [
            "😊 Excellent état émotionnel !",
            "👍 Continuez à maintenir cet équilibre",
            "🎯 Profitez de ce moment pour être productif",
            "🌟 Gardez cette sérénité"
        ],
        'stress': [
            "🧘 Pratiquez la respiration profonde (4-7-8)",
            "🎵 Écoutez de la musique relaxante",
            "🌳 Sortez prendre l'air frais",
            "☕ Faites une pause et déconnectez-vous",
            "🗣️ Parlez à quelqu'un de confiance"
        ]
    }

config = Config()

# ============================================
# 2. CHARGEMENT DU MODÈLE
# ============================================

def load_model():
    """Charge le modèle pré-entraîné avec gestion d'erreurs"""
    try:
        model_files = sorted(config.MODELS_DIR.glob("facestress_best_*.pth"))
        if not model_files:
            raise FileNotFoundError(
                f"Aucun modèle trouvé dans {config.MODELS_DIR}\n"
                "Assurez-vous d'avoir entraîné et sauvegardé un modèle."
            )

        model_path = model_files[-1]
        logger.info(f"Chargement du modèle: {model_path.name}")
        
        # Initialisation du modèle
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
        
        # Chargement des poids
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        model.eval()
        
        logger.info(f"Modèle chargé avec succès sur {config.DEVICE}")
        return model, checkpoint
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

# Initialisation du modèle
try:
    model, checkpoint = load_model()
    model_loaded = True
except Exception as e:
    logger.error(f"Impossible de charger le modèle: {e}")
    model_loaded = False
    model, checkpoint = None, None

# ============================================
# 3. TRANSFORMATION DES IMAGES
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# 4. FONCTION DE PRÉDICTION
# ============================================

def predict_stress_fatigue(image):
    """
    Prédit l'état émotionnel à partir d'une image
    
    Args:
        image: Image PIL ou numpy array
    
    Returns:
        tuple: (graphique, indice_stress, émotion_principale, conseils)
    """
    # Vérification du modèle
    if not model_loaded or model is None:
        error_msg = "⚠️ Le modèle n'est pas chargé. Veuillez vérifier votre installation."
        return None, "0", "⚪️ N/A", error_msg
    
    # Vérification de l'image
    if image is None:
        return None, "0", "⚪️ N/A", "💡 Uploadez une image pour obtenir une analyse."
    
    try:
        # Conversion de l'image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prédiction
        img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Traitement des résultats
        probs_dict = {
            config.CLASSES[i]: float(probabilities[i]) * 100 
            for i in range(config.NUM_CLASSES)
        }
        predicted_class = config.CLASSES[torch.argmax(probabilities).item()]
        stress_index = int(probs_dict.get('stress', 0))
        confidence = probs_dict[predicted_class]

        # Création du graphique amélioré
        fig = create_probability_chart(probs_dict)

        # Génération des sorties
        main_emotion_output = (
            f"{config.CLASS_INFO[predicted_class]['emoji']} "
            f"{config.CLASS_INFO[predicted_class]['label']} "
            f"({confidence:.1f}%)"
        )
        
        advice_list = config.CLASS_ADVICE.get(predicted_class, [])
        advice_output = (
            f"### 💡 Recommandations pour l'état: {config.CLASS_INFO[predicted_class]['label']}\n\n" +
            "\n".join([f"- {adv}" for adv in advice_list])
        )

        return fig, str(stress_index), main_emotion_output, advice_output
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        error_msg = f"❌ Erreur lors de l'analyse: {str(e)}"
        return None, "0", "⚪️ Erreur", error_msg

def create_probability_chart(probs_dict):
    """Crée un graphique en barres amélioré avec Plotly"""
    labels = [config.CLASS_INFO[c]['label'] for c in probs_dict.keys()]
    values = list(probs_dict.values())
    colors = [config.CLASS_INFO[c]['color'] for c in probs_dict.keys()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2)
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(size=14, color='#000000', family='Arial Black')
        )
    ])
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            title="Probabilité (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="",
            showgrid=False
        ),
        paper_bgcolor='rgba(255,255,255,0.7)',
        plot_bgcolor='rgba(255,255,255,0.3)',
        font=dict(color="#000000", size=12),
        margin=dict(l=10, r=50, t=10, b=10),
        height=250
    )
    
    return fig

# ============================================
# 5. INTERFACE GRADIO
# ============================================

def create_interface():
    """Crée l'interface Gradio avec style amélioré"""
    
    css = """
    :root {
        --primary-blue: #004eff;
        --secondary-cyan: #31afd4;
        --accent-pink: #ff007f;
        --text-dark: #050505;
        --bg-light: #fefefe;
    }

    .gradio-container { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--text-dark);
    }

    .panel { 
        background: rgba(255, 255, 255, 0.85) !important; 
        backdrop-filter: blur(20px); 
        border-radius: 20px; 
        padding: 20px; 
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .panel:hover { 
        transform: translateY(-8px); 
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }

    .gr-button { 
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-cyan)) !important;
        color: white !important; 
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 78, 255, 0.3) !important;
    }

    .gr-button:hover { 
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 78, 255, 0.4) !important;
    }

    .gr-textbox { 
        border-radius: 12px !important;
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    h1 {
        color: var(--text-dark) !important;
        font-size: 3em !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    h3 {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }

    .markdown-text {
        font-size: 16px;
        line-height: 1.8;
    }
    """

    with gr.Blocks(title="FaceStress AI Dashboard", theme=gr.themes.Soft(), css=css) as interface:
        
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #050505; font-size: 3.5em; font-weight: 900; margin-bottom: 10px; 
                           text-shadow: 3px 3px 6px rgba(0,0,0,0.15);">
                    🧠 FaceStress AI - Analyse Émotionnelle
                </h1>
                <h3 style="color: #333; font-size: 1.3em; font-weight: 500; margin-top: 10px;">
                    Détection intelligente du stress et de la fatigue par analyse faciale
                </h3>
                <p style="color: #666; font-size: 1.1em; margin-top: 15px;">
                    Uploadez une image ou utilisez votre webcam pour une analyse en temps réel
                </p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    type="pil",
                    label="📸 Importez une image ou utilisez la webcam",
                    sources=["upload", "webcam"],
                    elem_classes=["panel"]
                )
                
                advice_output = gr.Markdown(
                    "💡 **Les conseils personnalisés s'afficheront ici après l'analyse**",
                    elem_classes=["panel", "markdown-text"]
                )

            with gr.Column(scale=3):
                with gr.Row():
                    stress_index_output = gr.Textbox(
                        label="📊 Indice de Stress (%)",
                        interactive=False,
                        elem_classes=["panel"],
                        elem_id="stress-index"
                    )
                    main_emotion_output = gr.Textbox(
                        label="🎭 État Émotionnel Détecté",
                        interactive=False,
                        elem_classes=["panel"],
                        elem_id="main-emotion"
                    )
                
                probs_plot = gr.Plot(
                    label="📈 Distribution des Probabilités",
                    elem_classes=["panel"]
                )

        # Informations supplémentaires
        with gr.Accordion("ℹ️ À propos de FaceStress AI", open=False):
            gr.Markdown(
                """
                **FaceStress AI** utilise un réseau de neurones profond (MobileNetV2) 
                entraîné pour reconnaître trois états émotionnels:
                
                - 😴 **Fatigue**: Détecte les signes de fatigue physique et mentale
                - 😊 **Normal**: État émotionnel équilibré et serein
                - 😰 **Stress**: Identifie les marqueurs de stress et d'anxiété
                
                **Technologie**: PyTorch + MobileNetV2 + Transfer Learning
                """
            )

        # Événements
        image_input.change(
            fn=predict_stress_fatigue,
            inputs=[image_input],
            outputs=[probs_plot, stress_index_output, main_emotion_output, advice_output]
        )

    return interface

# ============================================
# 6. LANCEMENT
# ============================================

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            favicon_path=None
        )
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'interface: {e}")
        raise