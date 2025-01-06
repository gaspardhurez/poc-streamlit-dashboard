import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image
import pandas as pd
import json

# Configuration du thème
st.set_page_config(
    page_title="Dashboard Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le tokenizer et le modèle
@st.cache_resource
def load_model_and_tokenizer(model_dir="gaspardhurez/sentiment-analyzer-amazon"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# Charger les textes d'exemple
@st.cache_resource
def load_sample_texts():
    with open('sample_texts.json', 'r') as f:
        sample_texts = json.load(f)
    # Convertir en liste unique avec mention de la classe
    flat_texts = [{"text": text, "label": label} for label, texts in sample_texts.items() for text in texts]
    return flat_texts


# Effectuer une prédiction
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.detach().cpu().numpy()

# Interface utilisateur Streamlit
def main():
    # Barre de navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisissez une page :", ["Exploration des Données", "Analyse de Sentiment"])

    if page == "Exploration des Données":
        st.title("Exploration des Données (EDA)")
        st.write("""
        **Bienvenue sur la section EDA (Exploration de Données).** 
        Découvrez les statistiques clés et les visualisations pour mieux comprendre les données utilisées dans ce projet.
        """)

        # 1. Statistiques descriptives
        try:
            st.subheader("1. Statistiques Descriptives")
            describe_csv = "eda_plots/describe_statistics.csv"
            describe_df = pd.read_csv(describe_csv, index_col=0)
            st.dataframe(describe_df)
            st.download_button(
                label="Télécharger les Statistiques Descriptives",
                data=describe_df.to_csv().encode('utf-8'),
                file_name="describe_statistics.csv",
                mime="text/csv"
            )
        except FileNotFoundError:
            st.error("Le fichier describe_statistics.csv est introuvable.")

        # 2. Wordcloud des commentaires
        try:
            st.subheader("2. Wordcloud des Commentaires")
            wordcloud_img = Image.open("eda_plots/wordcloud_comments.png")
            st.image(wordcloud_img, caption="Wordcloud des commentaires", use_container_width=True)
        except FileNotFoundError:
            st.error("Le fichier wordcloud_comments.png est introuvable.")

        # 3. Longueur des commentaires (box plot)
        try:
            st.subheader("3. Longueur des Commentaires")
            length_boxplot_img = Image.open("eda_plots/comment_length_boxplot_iqr.png")
            st.image(
                length_boxplot_img,
                caption="Distribution de la Longueur des Commentaires (sans outliers)",
                use_container_width=True
            )
        except FileNotFoundError:
            st.error("Le fichier comment_length_boxplot_iqr.png est introuvable.")

        # 4. Histogramme des Ratings
        try:
            st.subheader("4. Histogramme des Ratings")
            rating_hist_img = Image.open("eda_plots/rating_distribution.png")
            st.image(
                rating_hist_img,
                caption="Histogramme des Ratings",
                use_container_width=True
            )
        except FileNotFoundError:
            st.error("Le fichier rating_distribution.png est introuvable.")
        
    elif page == "Analyse de Sentiment":
        st.title("Analyse de Sentiment avec Transformer")
        st.write("""
        **Entrez un texte pour analyser son sentiment.** 
        Vous pouvez également sélectionner un texte d'exemple dans la liste déroulante ci-dessous.
        """)

        # Chargement du modèle
        with st.spinner("Chargement du modèle..."):
            tokenizer, model = load_model_and_tokenizer()

        # Charger les textes d'exemple
        sample_texts = load_sample_texts()

        # Interface utilisateur pour l'entrée du texte
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Entrée manuelle")
            text_input = st.text_area(
                "Entrez un texte pour analyser son sentiment :",
                height=150,
                placeholder="Exemple : Ce produit est fantastique, je l'adore !"
            )

        with col2:
            st.write("### Sélectionner un texte d'exemple")
            selected_sample = st.selectbox(
                "Choisissez un texte d'exemple à analyser :",
                options=[""] + sample_texts,
                format_func=lambda x: "Sélectionnez un exemple" if x == "" else f"[{x['label']}] {x['text'][:75]}..." if len(x['text']) > 75 else f"[{x['label']}] {x['text']}"
            )

        # Déterminer le texte à analyser
        text_to_analyze = None
        if text_input.strip():
            text_to_analyze = text_input.strip()
        elif selected_sample and isinstance(selected_sample, dict):
            text_to_analyze = selected_sample["text"]

        # Bouton d'analyse
        if st.button("Analyser"):
            if text_to_analyze:
                with st.spinner("Analyse en cours..."):
                    predicted_class, probabilities = predict_sentiment(text_to_analyze, tokenizer, model)

                    # Mapping des classes
                    class_mapping = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                    sentiment = class_mapping[predicted_class]

                    # Affichage des résultats
                    st.success(f"**Sentiment prédit : {sentiment}**")
                    st.write("Probabilités associées :")
                    for idx, prob in enumerate(probabilities[0]):
                        st.write(f"{class_mapping[idx]} : {prob * 100:.2f}%")

                    # Afficher le texte analysé
                    st.write("**Texte analysé :**")
                    st.write(f"\"{text_to_analyze}\"")
            else:
                st.warning("Erreur : Veuillez entrer un texte ou sélectionner un exemple avant de lancer l'analyse.")

if __name__ == "__main__":
    main()
