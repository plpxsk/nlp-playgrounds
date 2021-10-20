import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def make_words(text):
    """http://norvig.com/spell-correct.html"""
    return re.findall(r'\w+', text.lower())


def get_words_counter(words, remove_stop=False):
    if remove_stop:
        words = [x for x in words if x.lower() not in ENGLISH_STOP_WORDS]
    c = Counter(words)
    return(c)


def get_data():
    with open("../abstracts_manual.txt") as f:
        x = f.read()
    x = x.split("\n\n")
    x = [i.replace("\n", " ") for i in x]
    return(x)


def run_model(vectors, method="svd", n_components=2):
    if method == "svd":
        U, s, Vh = linalg.svd(vectors, full_matrices=False)
        r1 = Vh
        r2 = U
    elif method == "nmf":
        clf = decomposition.NMF(n_components=n_components, random_state=1)
        W1 = clf.fit_transform(np.asarray(vectors))
        H1 = clf.components_
        r1 = H1
        r2 = W1
    elif method == "tfidf":
        clf = decomposition.NMF(n_components=n_components, random_state=1,
                                max_iter=800)
        vectorizer_tfidf = TfidfVectorizer(stop_words='english')
        vectors_tfidf = vectorizer_tfidf.fit_transform(abstracts)
        W1 = clf.fit_transform(vectors_tfidf)
        H1 = clf.components_
        r1 = H1
        r2 = W1
    return(r1, r2)


def show_topics(a, num_top_words=8):
    def top_words(t): return [vocab[i]
                              for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]


def get_members_for_cluster(W1, i):
    cluster_membership_index = np.argmax(W1, axis=1)
    members = np.where(cluster_membership_index == i)
    members = np.array(members).flatten()
    return(members)


def add_abstract_viewer(abstracts, key):
    with st.expander('View abstracts'):
        # Show ONE at a time only. Otherwise, with st.multiselect, they could
        # appear out of order
        s = st.selectbox("Select:", range(len(abstracts)), key=key)
        st.write(abstracts[s])


def main(abstracts, vectors):

    st.title("MedLP")
    st.markdown("""Find topic clusters among medical abstracts""")

    st.markdown("""## Abstracts""")
    ll = len(abstracts)
    "Found # of abstracts:"
    ll
    add_abstract_viewer(abstracts, 1)

    st.markdown("## Most common words")
    k = st.slider("Show top K words in abstract: ", 2, 100, 8)
    remove_stop = st.checkbox('Remove stop words', value=True)
    text = "".join(abstracts)
    words = make_words(text)
    c = get_words_counter(words, remove_stop=remove_stop)
    top = c.most_common(k)
    top = pd.DataFrame(top, columns=['Word', 'N']).set_index('Word')
    st.write(top)

    st.markdown("## Converts abstracts into topics")
    st.markdown(
        "Extracts topics for each abstract with `SVD`. Shows 'strongest'(?) topics first")  # noqa
    num_top_words = st.slider("Number of top words: ", 2, 100, 8)
    sel = st.slider("Display strongest (?) K topics:", 1, len(abstracts), 3)
    m, _ = run_model(vectors, "svd")
    r = show_topics(m[:sel], num_top_words)
    r

    st.markdown("## Cluster abstracts")
    n_clusters = st.slider("Select number of clusters. Use `1` for top topics among entire corpus.",  # noqa
                           1, len(abstracts), value=3)
    st.markdown("### NMF")
    H1, W1 = run_model(vectors, "nmf", n_clusters)
    r = show_topics(H1, num_top_words)
    r
    st.markdown("Members:")
    for i in range(n_clusters):
        m = get_members_for_cluster(W1, i)
        st.write(f"Cluster {i}: {m}")
    add_abstract_viewer(abstracts, 2)

    st.markdown("### TF-IDF")
    H1, W1 = run_model(vectors, "tfidf", n_clusters)
    r = show_topics(H1, num_top_words)
    r
    st.markdown("Members:")
    for i in range(n_clusters):
        m = get_members_for_cluster(W1, i)
        st.write(f"Cluster {i}: {m}")
    add_abstract_viewer(abstracts, 3)


abstracts = get_data()
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(abstracts).todense()
vocab = np.array(vectorizer.get_feature_names_out())


main(abstracts=abstracts, vectors=vectors)
