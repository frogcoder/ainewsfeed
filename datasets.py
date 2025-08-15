import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine


DEFAULT_CONNECTION_STRING = "postgresql://newsfeed:newsfeed@localhost:5432/newsfeed"

def load_ratings(
        connection_string: str=DEFAULT_CONNECTION_STRING,
        max_rating: int=5
) -> tf.data.Dataset:
    query = """
    SELECT user_id, embedding_id, COUNT(*) AS rating FROM
    (SELECT o.user_id, o.embedding_id
     FROM opened_articles o
     JOIN articles a ON o.embedding_id=a.embedding_id
     UNION
     SELECT c.user_id, c.embedding_id
     FROM user_comments c
     join articles a ON c.embedding_id=a.embedding_id) ratings
    GROUP BY user_id, embedding_id
    """
        
    engine = create_engine(connection_string)
    df = pd.read_sql_query(query, engine)

    return tf.data.Dataset.from_tensor_slices({
        "user_id": df["user_id"].values,
        "embedding_id": df["embedding_id"].values,
        "rating": df["rating"].where(df["rating"] <= max_rating, max_rating).values
    })


def load_articles(
        connection_string=DEFAULT_CONNECTION_STRING) -> tf.data.Dataset:
    query = "SELECT embedding_id, title, source from articles WHERE embedding_id IS NOT NULL"
    engine = create_engine(connection_string)
    df = pd.read_sql_query(query, engine)

    return tf.data.Dataset.from_tensor_slices({
        "embedding_id": df["embedding_id"].values,
        "title": df["title"].values,
        "source": df["source"].values
    })
