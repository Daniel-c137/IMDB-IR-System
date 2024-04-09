from typing import Dict, List
from .core.search import SearchEngine
from .core.spell_correction import SpellCorrection
from .core.snippet import Snippet
from .core.indexer.indexes_enum import Indexes, Index_types
import json

movies_dataset = {}
with open('IMDB_crawled.json', 'r') as f:
    j = json.load(f)
    for doc in j:
        movies_dataset[doc['id']] = [doc]
    f.close()
search_engine = SearchEngine()

documents = None
with open('IMDB_crawled_pre_processed.json', 'r') as f:
    documents = json.load(f)
    for doc in documents:
        movies_dataset[doc['id']].append(doc)
    f.close()
s = SpellCorrection(documents)

def correct_text(text: str) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # any preprocessing steps here
    spell_correction_obj = SpellCorrection(documents)
    text = spell_correction_obj.spell_check(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weight_stars, weight_genres, weight_summary = weights
    weights = {
        Indexes.STARS: weight_stars,
        Indexes.GENRES: weight_genres,
        Indexes.SUMMARIES: weight_summary
    }
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    result, processed_result = movies_dataset.get(
        id,
        ({
            "Title": "This is movie's title",
            "Summary": "This is a summary",
            "URL": "https://www.imdb.com/title/tt0111161/",
            "Cast": ["Morgan Freeman", "Tim Robbins"],
            "Genres": ["Drama", "Crime"],
            "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
        }, None),
    )

    result["Image_URL"] = (
        result['Image_URL']
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result, processed_result
