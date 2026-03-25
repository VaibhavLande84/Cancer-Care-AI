from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever

def get_docs(query):
    retriever = WikipediaRetriever(top_k_results=2, lang="en")
    query = query
    docs_wiki = retriever.invoke(query)

    retriever = ArxivRetriever(load_max_docs=8)
    docs_research = retriever.invoke(query)
    return docs_wiki, docs_research
