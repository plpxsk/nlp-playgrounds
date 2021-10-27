import spacy
import pandas as pd
import streamlit as st
from spacy_streamlit import visualize_ner
from scispacy.abbreviation import AbbreviationDetector


def load_data():
    with open("abstracts_manual.txt") as f:
        x = f.read()
    x = x.replace("\n\n", " ")
    x = x.replace("\n", " ")
    return(x)


@st.cache(allow_output_mutation=True)
def load_model(name):
    """Adapted from https://gist.github.com/DeNeutoy/b20860b40b9fa9d33675893c56afde42#file-app-py-L16"""
    nlp = spacy.load(name)
    # Add abbreviation detector
    nlp.add_pipe('abbreviation_detector')
    return nlp


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    """https://gist.github.com/DeNeutoy/b20860b40b9fa9d33675893c56afde42#file-app-py-L25"""
    nlp = load_model(model_name)
    return nlp(text)


def get_special_entities(doc):
    """https://gist.github.com/DeNeutoy/b20860b40b9fa9d33675893c56afde42#file-app-py-L121"""
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
    ]
    return(data, attrs)


def get_abbrevs(doc, nlp) -> dict:
    abbrevs = dict()
    for abrv in doc._.abbreviations:
        abbrevs[str(abrv)] = str(abrv._.long_form)
    return(abbrevs)


TEXT = load_data()

st.sidebar.title("MedLP")
spacy_model = st.sidebar.selectbox("Select trained NLP model",
                                   ["en_core_sci_md", "en_core_web_sm",
                                    "en_ner_bionlp13cg_md"])
# load model selected in sidebar
nlp = load_model(spacy_model)

visualizers = st.sidebar.multiselect(
    "Select analyses",
    ["abbrev", "special", "genes", "ner"], default=["abbrev", "genes"])


st.header("Load text data")
st.markdown("Default text is 18 cancer paper abstracts")
text = st.text_area("You can paste your own text here:", TEXT)
doc = process_text(spacy_model, text)


if 'abbrev' in visualizers:
    st.header("Abbreviations")
    x = get_abbrevs(doc, nlp)
    x = pd.Series(x, name='Abbreviation').sort_index()
    st.write(x)

if 'genes' in visualizers:
    st.header("Genes")
    kind = ["GENE_OR_GENE_PRODUCT"]
    u1 = st.checkbox("Filter to unique tokens?", value=False, key=1)
    df, attrs = get_special_entities(doc)
    df = pd.DataFrame(df, columns=attrs)
    df = df[df['label_'].isin(kind)]
    df = df.drop('label_', axis=1)
    if u1:
        df = df.drop_duplicates('text')
    st.write(df)


if 'special' in visualizers:
    st.header("Special Entities")
    st.markdown("See **ner** for nice tagging.")
    entity_kinds = st.multiselect("Select entity kinds",
                                  nlp.get_pipe('ner').labels)

    u2 = st.checkbox("Filter to unique tokens?", value=False, key=2)
    df, attrs = get_special_entities(doc)
    df = pd.DataFrame(df, columns=attrs)
    df = df[df['label_'].isin(entity_kinds)]
    if u2:
        df = df.drop_duplicates('text')
    st.write(df)


if 'ner' in visualizers:
    st.header("Named Entity Recognition")
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title=None)
