import spacy
from spacy_streamlit import load_model, process_text, visualize_ner
import pandas as pd
import streamlit as st


def load_data():
    with open("abstracts_manual.txt") as f:
        x = f.read()
    x = x.replace("\n\n", " ")
    x = x.replace("\n", " ")
    return(x)


def get_special_entities(doc):
    # https://gist.github.com/DeNeutoy/b20860b40b9fa9d33675893c56afde42#file-app-py-L121
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
    ]
    return(data, attrs)


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

TEXT = load_data()

st.sidebar.title("MedLP")
spacy_model = st.sidebar.selectbox("Select trained NLP model",
                                   ["en_core_sci_sm", "en_core_web_sm",
                                    "en_ner_bionlp13cg_md"])
# load model selected in sidebar
nlp = load_model(spacy_model)

visualizers = st.sidebar.multiselect("Select analyses", ["abbrev", "special", "genes", "ner"],
                                     default=["abbrev", "special"])


entity_kinds = st.sidebar.multiselect("Select entity kinds", nlp.get_pipe('ner').labels,
                                      default=['GENE_OR_GENE_PRODUCT'])


st.title("Load text data")
text = st.text_area("Text to analyze", TEXT)
doc = process_text(spacy_model, text)


if 'abbrev' in visualizers:
    st.title("Abbreviations")
    x = get_abbrevs(doc, nlp)
    x = pd.Series(x, name='Abbreviation')
    st.write(x)

if 'special' in visualizers:
    st.title("Special Entities")
    u = st.checkbox("Filter to unique?", value=False)

    df, attrs = get_special_entities(doc)
    df = pd.DataFrame(df, columns=attrs)
    df = df[df['label_'].isin(entity_kinds)]

    st.write(df)


if 'genes' in visualizers:
    st.title("Genes")


if 'ner' in visualizers:
    st.title("Named Entity Recognition")
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
