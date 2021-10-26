import spacy
from spacy_streamlit import load_model, process_text, visualize_ner
import pandas as pd
import streamlit as st


def get_abbrevs(doc, nlp) -> dict:
    #from scispacy.abbreviation import AbbreviationDetector
    # nlp.add_pipe("abbreviation_detector")
    abbrevs = dict()
    for abrv in doc._.abbreviations:
        abbrevs[str(abrv)] = str(abrv._.long_form)
    return(abbrevs)


TEXT = "Sundar Pichai  is the CEO of Google."

TEXT = "Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily."

TEXT = "PD1 and CTLA4 are dynamically expressed on different T cell subsets that can either disrupt or sustain tumor growth."

st.sidebar.title("MedLP")
spacy_model = st.sidebar.selectbox(
    "Select trained NLP model", ["en_core_sci_sm", "en_core_web_sm"])
visualizers = st.sidebar.multiselect("Select analyses", ["ner", "abbrev"],
                                     default=["abbrev"])


# load model selected in sidebar
nlp = load_model(spacy_model)

text = st.text_area("Text to analyze", TEXT)
doc = process_text(spacy_model, text)

if 'ner' in visualizers:
    st.title("Named Entity Recognition")
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)


if 'abbrev' in visualizers:
    st.title("Abbreviations")
    a = get_abbrevs(doc, nlp)
    a = pd.Series(a, name='Abbreviation')
    st.write(a)
