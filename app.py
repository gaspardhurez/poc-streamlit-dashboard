import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le tokenizer et le modèle
@st.cache_resource
def load_model_and_tokenizer(model_dir="gaspardhurez/sentiment-analyzer-amazon"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# Effectuer une prédiction
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.detach().cpu().numpy()

# Interface utilisateur Streamlit
def main():
    st.title("Analyse de Sentiment avec Transformer")
    st.write("Ce modèle analyse le sentiment des textes en utilisant un Transformer fine-tuné.")

    # Chargement du modèle
    with st.spinner("Chargement du modèle..."):
        tokenizer, model = load_model_and_tokenizer()

    # Entrée utilisateur
    text_input = st.text_area("Entrez un texte pour analyser son sentiment :", height=150)
    
    if st.button("Analyser"):
        if text_input.strip():
            with st.spinner("Analyse en cours..."):
                predicted_class, probabilities = predict_sentiment(text_input, tokenizer, model)

                # Mapping des classes
                class_mapping = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                sentiment = class_mapping[predicted_class]

                # Affichage des résultats
                st.success(f"Sentiment prédit : {sentiment}")
                st.write("Probabilités associées :")
                for idx, prob in enumerate(probabilities[0]):
                    st.write(f"{class_mapping[idx]} : {prob * 100:.2f}%")
        else:
            st.warning("Veuillez entrer un texte avant de lancer l'analyse.")

if __name__ == "__main__":
    main()