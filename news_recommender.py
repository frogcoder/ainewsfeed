import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, Text, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsFeedDataPipeline:
    """Data pipeline for loading and preprocessing newsfeed data from PostgreSQL"""
    
    def __init__(self, db_connection_string: str):
        """
        Initialize data pipeline
        
        Args:
            db_connection_string: PostgreSQL connection string
                Format: "postgresql://user:password@host:port/database"
        """
        self.engine = create_engine(db_connection_string)
        
    def load_all_articles(self) -> pd.DataFrame:
        query = """ SELECT title FROM articles """
        df = pd.read_sql_query(query, self.engine)
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load data from PostgreSQL and calculate ratings"""
        
        # SQL query to join all tables and calculate ratings
        query = """
        WITH user_article_stats AS (
            SELECT
                nu.id,
                nu.user_id,
                a.id AS article_id,
                a.embedding_id,
                a.title,
                a.published_date,
                COUNT(DISTINCT oa.id) as open_count,
                COUNT(DISTINCT uc.id) as comment_count
            FROM newsfeed_users nu
            CROSS JOIN articles a
            LEFT JOIN opened_articles oa ON nu.user_id = oa.user_id AND a.embedding_id = oa.embedding_id
            LEFT JOIN user_comments uc ON nu.user_id = uc.user_id AND a.embedding_id = uc.embedding_id
            GROUP BY nu.id, nu.user_id, a.id, a.embedding_id, a.title, a.published_date
        ),
        ratings_calculated AS (
            SELECT
                id,
                user_id,
                article_id,
                embedding_id,
                title,
                published_date,
                open_count,
                comment_count,
                LEAST(
                    5.0, 
                    GREATEST(
                        0.0,
                        (open_count * 1.0) + (comment_count * 2.0)  -- Weight comments more heavily
                    )
                ) as rating
            FROM user_article_stats
            WHERE open_count > 0 OR comment_count > 0  -- Only include interactions
        )
        SELECT * FROM ratings_calculated
        ORDER BY user_id, embedding_id;
        """
        
        logger.info("Loading data from PostgreSQL...")
        df = pd.read_sql_query(query, self.engine)
        
        # Add user features (this could be extended with more user metadata)
        user_features_query = """
        SELECT 
            nu.user_id,
            COUNT(DISTINCT oa.embedding_id) as total_articles_opened,
            COUNT(DISTINCT uc.embedding_id) as total_articles_commented,
            AVG(EXTRACT(HOUR FROM oa.opened_at)) as avg_reading_hour
        FROM newsfeed_users nu
        LEFT JOIN opened_articles oa ON nu.user_id = oa.user_id
        LEFT JOIN user_comments uc ON nu.user_id = uc.user_id
        GROUP BY nu.user_id;
        """
        
        user_features = pd.read_sql_query(user_features_query, self.engine)
        df = df.merge(user_features, on='user_id', how='left')
        
        # Fill NaN values
        df = df.fillna(0)
        
        logger.info(f"Loaded {len(df)} interaction records")
        return df

class TwoTowerNewsRecommender(tfrs.Model):
    """Two-Tower Recommender Model for News Articles"""
    
    def __init__(self, 
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0,
                 embedding_dimension: int = 64,
                 user_vocab: Optional[tf.data.Dataset] = None,
                 article_vocab: Optional[tf.data.Dataset] = None):
        super().__init__()
        
        self.embedding_dimension = embedding_dimension
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        
        # User and article vocabularies
        self.user_vocab = user_vocab
        self.article_vocab = article_vocab

        print("user vocabularies:", user_vocab, "size: ", user_vocab.size, user_vocab.dtype)
        # User embedding tower
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=user_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(user_vocab) + 1, embedding_dimension)
        ])
        
        # Article embedding tower
        self.article_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=np.array(list(article_vocab.as_numpy_iterator())), mask_token=None),
            tf.keras.layers.Embedding(len(article_vocab) + 1, embedding_dimension)
        ])
        
        # Additional user features
        self.user_features_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        
        # Rating prediction
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        metrics = tfrs.metrics.FactorizedTopK(
            candidates=article_vocab.batch(128).map(self.article_embedding)
        )
        
        # Retrieval task
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=metrics
        )
        
        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
    
    def call(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # User embeddings
        user_emb = self.user_embedding(features["user_id"])
        
        # User features (numerical)
        user_features = tf.stack([
            tf.cast(features["total_articles_opened"], tf.float32),
            tf.cast(features["total_articles_commented"], tf.float32),
            tf.cast(features["avg_reading_hour"], tf.float32)
        ], axis=1)
        user_feature_emb = self.user_features_embedding(user_features)
        
        # Combine user embeddings
        user_emb = user_emb + user_feature_emb
        
        # Article embeddings
        article_emb = self.article_embedding(features["embedding_id"])
        
        return {
            "user_embedding": user_emb,
            "article_embedding": article_emb,
            "predicted_rating": self.rating_model(
                tf.concat([user_emb, article_emb], axis=1)
            )
        }
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_features_embedding(features["user_id"])
        positive_article_embeddings = self.article_embedding(features["article_id"])
        predicted_rating = self(features)["predicted_rating"]
        
        # Retrieval loss
        retrieval_loss = self.retrieval_task(
            user_embeddings,
            positive_article_embeddings
        )
        return retrieval_loss


class NewsRecommenderSystem:
    """Complete recommender system with training and inference"""
    
    def __init__(self, db_connection_string: str):
        self.data_pipeline = NewsFeedDataPipeline(db_connection_string)
        self.model = None
        self.train_ds = None
        self.test_ds = None
        
    def prepare_datasets(self, test_ratio: float = 0.2):
        """Prepare training and testing datasets"""
        # Load data
        df = self.data_pipeline.load_and_prepare_data()
        articles_df = self.data_pipeline.load_all_articles()
        
        # Convert to tensorflow dataset format
        ratings_ds = tf.data.Dataset.from_tensor_slices({
            "user_id": df["user_id"].astype(str).values,
            "article_id": df["article_id"].astype(str).values,
            "embedding_id": df["embedding_id"].astype(str).values,
            "rating": df["rating"].astype(np.float32).values,
            "total_articles_opened": df["total_articles_opened"].astype(np.float32).values,
            "total_articles_commented": df["total_articles_commented"].astype(np.float32).values,
            "avg_reading_hour": df["avg_reading_hour"].astype(np.float32).values,
        })
        
        # Shuffle and split
        shuffled = ratings_ds.shuffle(100000, seed=42, reshuffle_each_iteration=False)
        
        total_size = len(df)
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size
        
        self.train_ds = shuffled.skip(test_size).batch(8192).cache()
        self.test_ds = shuffled.take(test_size).batch(4096).cache()
        
        # Create vocabularies
        feature_ds = ratings_ds.map(lambda x: {
            "user_id": x["user_id"],
            "embedding_id": x["embedding_id"],
            "article_id": x["article_id"]
        })
        
        user_ids = feature_ds.map(lambda x: x["user_id"])
        #article_ids = feature_ds.map(lambda x: x["article_id"])
        article_ids = articles_df["title"]
        
        self.user_vocab = np.array(list(user_ids.unique().as_numpy_iterator()))
        self.article_vocab = tf.data.Dataset.from_tensor_slices(article_ids.unique())
        
        logger.info(f"Prepared {train_size} training and {test_size} test samples")
        
    def build_model(self, embedding_dimension: int = 64):
        """Build the two-tower model"""
        print("User vocabularies:", self.user_vocab)
        self.model = TwoTowerNewsRecommender(
            rating_weight=1.0,
            retrieval_weight=1.0,
            embedding_dimension=embedding_dimension,
            user_vocab=self.user_vocab,
            article_vocab=self.article_vocab
        )
        
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        
    def train(self, epochs: int = 3):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        logger.info("Starting model training...")
        
        # Train the model
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            validation_data=self.test_ds,
            verbose=1
        )
        
        logger.info("Training completed!")
        
    def evaluate(self):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        metrics = self.model.evaluate(self.test_ds, return_dict=True)
        logger.info(f"Test metrics: {metrics}")
        return metrics
        
    def get_recommendations(self, user_id: str, k: int = 10):
        """Get top-k recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Create a retrieval model for inference
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_embedding)
        index.index_from_dataset(
            tf.data.Dataset.zip((
                self.article_vocab,
                self.article_vocab.map(self.model.article_embedding)
            ))
        )
        
        # Get recommendations
        _, top_candidate_indices = index(tf.constant([user_id]))
        
        return top_candidate_indices[0, :k].numpy()
        
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Database configuration
    DB_CONNECTION = "postgresql://newsfeed:newsfeed@localhost:5432/newsfeed"
    
    # Initialize recommender system
    recommender = NewsRecommenderSystem(DB_CONNECTION)
    
    # Prepare data
    recommender.prepare_datasets(test_ratio=0.2)
    
    # Build model
    recommender.build_model(embedding_dimension=64)
    
    # Train model
    recommender.train(epochs=5)
    
    # Evaluate model
    metrics = recommender.evaluate()
    print(f"Final metrics: {metrics}")
    
    # Get recommendations for a user
    try:
        recommendations = recommender.get_recommendations("user_123", k=10)
        print(f"Top 10 recommendations for user_123: {recommendations}")
    except Exception as e:
        print(f"Error getting recommendations: {e}")
    
    # Save model
    recommender.save_model("./news_recommender_model")
