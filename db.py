import os
from dataclasses import dataclass
from datetime import datetime
import psycopg2
import newsscrapper
from datasets import CHROMA_PATH
from datasets import POSTGRES_URL


db_manager = newsscrapper.get_chroma_data_manager(
    os.getenv("CHROMA_PATH", CHROMA_PATH)
)


@dataclass
class Article:
    title: str
    url: str
    published_date: datetime
    author: str
    summary: str
    source: str
    embedding_id: str


def query_articles(embedding_ids):
    sql = "SELECT title, url, published_date, author, summary, source, embedding_id FROM articles WHERE embedding_id IN %s"
    ids = tuple(embedding_ids)
    print("Embedding IDs", ids)
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ids,))
            articles = cur.fetchall()
            return [Article(*a) for a in articles]


def load_summary(embedding_id: str):
    sql = "SELECT summary FROM articles WHERE embedding_id=%s"
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_id,))
            summary = cur.fetchone()
            return summary[0]
    

def search_related_articles(embedding_id: str, max_count: int = 10):
    query = load_summary(embedding_id)

    print("Query found", query)

    """Search articles in ChromaDB"""
    try:
        results = db_manager.collection.query(
            query_texts=[query],
            n_results=max_count
        )
            
        search_results = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            search_results.append({
                'title': metadata['title'],
                'content': doc[:500] + "..." if len(doc) > 500 else doc,
                'url': metadata['url'],
                'source': metadata['source'],
                'author': metadata['author'],
                'published_date': metadata['published_date']
            })
            
        print("related article result", search_results)
        return search_results
            
    except Exception as e:
        print(f"Error searching articles: {e}")
        return []


def create_user(user_id: str):
    sql_user = "INSERT INTO newsfeed_users(user_id) VALUES(%s)"
    sql_opened = """
    INSERT INTO opened_articles (user_id, embedding_id)
    SELECT %s, embedding_id
    FROM (
    SELECT embedding_id,
        ROW_NUMBER() OVER (PARTITION BY source ORDER BY id) as rn
    FROM articles
    WHERE embedding_id IS NOT null
    ORDER BY scraped_at DESC
    ) ranked
    WHERE rn = 1
    """

    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_user, (user_id,))
            cur.execute(sql_opened, (user_id,))
    
