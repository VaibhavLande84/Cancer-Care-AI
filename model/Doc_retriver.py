from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever


def get_docs(query):
    """Retrieve Wikipedia and Arxiv docs. Returns ([], []) on any failure."""
    try:
        retriever = WikipediaRetriever(top_k_results=2, lang="en")
        docs_wiki = retriever.invoke(query)
    except Exception:
        docs_wiki = []

    try:
        retriever = ArxivRetriever(load_max_docs=8)
        docs_research = retriever.invoke(query)
    except Exception:
        docs_research = []

    return docs_wiki, docs_research

