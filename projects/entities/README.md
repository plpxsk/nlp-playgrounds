# Extract medical entities from text

From given text, app extracts and/or visualizes:

  * gene names
  * abbreviations
  * other user-selected entities such as tissue, cancer terms,

# App

Uses streamlit.io

Preview locally with `make prev`.

Install with `pip install -r requirements.txt`.


# References

Uses `scispacy` via [allenai/scispacy](https://github.com/allenai/scispacy)

And spacy-streamlit visualizer: [explosion/spacy-streamlit](https://github.com/explosion/spacy-streamlit)
