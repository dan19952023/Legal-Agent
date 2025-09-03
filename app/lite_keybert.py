import logging
import openai
from typing import List, Union, Tuple 
from operator import itemgetter
import itertools
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.engine import embed as embed_text, get_active_collection
from app.concurrency import sync2async
import asyncio

logger = logging.getLogger(__name__)

def max_sum_distance(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int,
    nr_candidates: int,
    threshold: float,
) -> List[Tuple[str, float]]:
    """Calculate Max Sum Distance for extraction of keywords.

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and
    extract the combination that are the least similar to each other
    by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large `top_n`

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception("Make sure that the number of candidates exceeds the number " "of keywords to return.")
    elif top_n > len(words):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [(words_vals[idx], round(float(distances[0][words_idx[idx]]), 4)) for idx in candidate if distances[0][words_idx[idx]] >= threshold]



def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    """
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # Extract and sort keywords in descending similarity
    keywords = [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]
    keywords = sorted(keywords, key=itemgetter(1), reverse=True)
    keywords = [k for k in keywords if k[1] >= threshold]
    return keywords

def merge_keywords(keywords: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Merge keywords that are similar to each other.
    """
    keywords = sorted(keywords, key=lambda x: len(x[0]), reverse=True)
    results = []

    for keyword, score in keywords:
        if any((keyword in kw) for kw, score in results):
            continue

        results.append((keyword, score))

    return results

async def batching_embed(texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = 8
    tasks = []
    active_collection = get_active_collection()

    if active_collection is None:
        logger.error("No active collection; Returning empty results")
        return np.array([]), np.array([])

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tasks.append(embed_text(batch, active_collection.model_id))

    index_map, embeddings = [], []

    for i, task in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
        if isinstance(task, Exception) or task is None:
            logger.error(f"Error embedding text: {task}")
            continue

        embeddings.extend(task)
        index_map.extend([i * batch_size + j for j in range(len(task))])

    if not embeddings:
        return np.array([]), np.array([])
        
    return np.array(embeddings), np.array(index_map)

class KeyBERT:
    """A minimal method for keyword extraction with BERT.

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with BERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.

    <div class="excalidraw">
    --8<-- "docs/images/pipeline.svg"
    </div>
    """

    def __init__(
        self
    ):
        """KeyBERT initialization.

        Arguments:
            model: Use a custom embedding model or a specific KeyBERT Backend.
                   The following backends are currently supported:
                      * SentenceTransformers
                      * 🤗 Transformers
                      * Flair
                      * Spacy
                      * Gensim
                      * USE (TF-Hub)
                    You can also pass in a string that points to one of the following
                    sentence-transformers models:
                      * https://www.sbert.net/docs/pretrained_models.html
            llm: The Large Language Model used to extract keywords
        """

    async def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        threshold: float = 0.5,
        merge: bool = False,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases.

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.
                           NOTE: when multiple documents are passed,
                           `seed_keywords`funtions in either of the two ways below:
                           - globally: when a flat list of str is passed, keywords are shared by all documents,
                           - locally: when a nested list of str is passed, keywords differs among documents.
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.
            threshold: Minimum similarity value between 0 and 1 used to decide how similar documents need to receive the same keywords.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(doc)
        ```

        To extract keywords from multiple documents, which is typically quite a bit faster:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(docs)
        ```
        """

        # Check for a single, empty document
        if isinstance(docs, str):
            docs = [docs]

        if not docs:
            return []

        vectorizer = CountVectorizer(
            ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            vocabulary=candidates,
        )
        
        try:
            count = vectorizer.fit(docs)
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            return []

        words = count.get_feature_names_out()

        if len(words) == 0:
            return []

        df = count.transform(docs)
        n_docs = len(docs)

        embeddings, index_map = await batching_embed([*docs, *words])

        if len(embeddings) == 0:
            logger.error("No embeddings generated; Returning empty results")
            return []

        doc_indices = np.where(index_map < n_docs)[0]
        docs = [docs[i] for i in doc_indices]

        word_indices = np.where(index_map >= n_docs)[0]
        words = [words[i - n_docs] for i in word_indices]

        doc_embeddings  = embeddings[:len(doc_indices)]
        word_embeddings = embeddings[len(doc_indices):]

        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):
            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                        threshold,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                        threshold,
                    )

                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]
                    keywords = [k for k in keywords if k[1] >= threshold]

                if merge:
                    keywords = merge_keywords(keywords)

                all_keywords.append(keywords)

            except ValueError:
                all_keywords.append([])

        return all_keywords
