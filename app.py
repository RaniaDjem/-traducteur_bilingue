import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Caching pour éviter de recharger les modèles à chaque fois
@st.cache_resource
def load_model(src_lang):
    if src_lang == "fr":
        model_name = "Helsinki-NLP/opus-mt-fr-en"
    else:
        model_name = "Helsinki-NLP/opus-mt-en-fr"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Fonction de traduction
def translate(text, src_lang):
    tokenizer, model = load_model(src_lang)
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translation = model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)


#Ignorer les erreurs de surveillance
#st.set_option("logger.level", "error")

# Interface Streamlit
st.set_page_config(page_title="Traducteur Bilingue 🇬🇧↔🇫🇷", layout="centered")
st.title("📘 Traducteur Bilingue")
st.markdown("Traduisez entre l'anglais 🇬🇧 et le français 🇫🇷 en un clic.")

# Sélection de la langue source
src_lang = st.selectbox("Langue du texte source :", ["fr", "en"], format_func=lambda x: "Français" if x=="fr" else "Anglais")

# Entrée de texte
text_input = st.text_area("Entrez le texte à traduire :", height=150)

# Traduction
if st.button("Traduire"):
    if text_input.strip():
        with st.spinner("Traduction en cours..."):
            result = translate(text_input, src_lang)
            st.success("Traduction :")
            st.text_area("Texte traduit :", result, height=150)
    else:
        st.warning("Veuillez entrer du texte à traduire.")
