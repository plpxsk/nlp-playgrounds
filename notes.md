# NLP Notes

Subjective and non-representative. Based on
[fast.ai](https://www.fast.ai/2019/07/08/fastai-nlp/).

# Topic modeling

## Takeaways

Decompose a term-document matrix to find neat properties, such as clusters of
topics among a bunch of documents (a corpus).

Essentially, use "math" to convert a "raw" representation of some text into
more useful constructs that can provide useful insights. This is matrix
"representation" (?).

TF-IDF seems like a good choice to find clusters since it's easy to interpret:
find topics in documents by using term freq scaled by inverse doc frequency.

NMF, SVD are other popular methods to decompose a matrix M:

M = U * S * Vt

S is a diagonal "scale" matrix which shows "strength" or "importance" of the
topic.

## Notes

Term-document matrix shows counts of terms in documents

Can "decompose" this matrix to find neat properties. Can decompose with NMF, SVD
