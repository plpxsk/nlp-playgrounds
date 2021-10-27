# NLP Notes

Subjective and non-representative. Based on
[fast.ai](https://www.fast.ai/2019/07/08/fastai-nlp/).

# Topic modeling

## Takeaways

Decompose a term-document matrix to find neat properties, such as clusters of
topics among a bunch of documents (a corpus).

Essentially, use "math" to convert a "raw" representation of some text into
more useful constructs that can provide useful insights.

This is "embedding" or matrix "representation".

TF-IDF seems like a good choice to find clusters since it's easy to interpret:
find topics in documents by using term freq scaled by inverse doc frequency.

SVD is a popular methods to decompose a matrix M:

raw matrix = features * scale * 

M = U * S * Vt

S is a diagonal "scale" matrix which shows "strength" or "importance" of the
topic.

NMF:

raw matrix = features * importances


## Concepts

Term-document matrix shows counts of terms in documents

Can "decompose" or "factorize" this matrix to find neat properties.

Factorization is like finding factors (elements) in multiplication (144 = 12 *
12 = 2 * 2 * 3 *...). You are decomposing something into components.

SVD, NMF: methods to decompose or factorize a matrix to find those properties.

  * SVD is very mathy and "exact" but harder to interpret, since it has
    negative values. Eg, negative value for a given word in a topic. What does
    that mean?
  * NMF provides better "interpretability" by eliminating negative values. It
    is not exact, and non unique.

SVD calculates all topics at once, whereas in NMF and truncated SVD, you
pre-specify the number of topics (n_components) as a hyper-parameter

This is for efficiency: don't need to calculate "less" important topics

Remember: these are kind of unsupervised clustering methods

Applications

  * SVD: Latent semantic analysis
  * NMF: facial and image recognition


### Also: Why Randomized algorithms?

Randomized, in-exact and perhaps non-unique algorithms may have advantages
especially of speed.

  * handles missing/incomplete data more efficiently
  * enables parallelization with GPUs







# References

Fast AI course videos (Youtube) and notes (github jupyter)

Intuitive vides of linear algrebra: 3Blue 1Brown and isomorphismes
