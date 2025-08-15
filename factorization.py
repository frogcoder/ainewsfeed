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
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load data from PostgreSQL and calculate ratings"""
        
        # SQL query to join all tables and calculate ratings
        query = """
        WITH user_article_stats AS (
            SELECT 
                nu.user_id,
                a.embedding_id,
                a.title,
                a.published_date,
                COUNT(DISTINCT oa.id) as open_count,
                COUNT(DISTINCT uc.id) as comment_count
            FROM newsfeed_users nu
            CROSS JOIN articles a
            LEFT JOIN opened_articles oa ON nu.user_id = oa.user_id AND a.embedding_id = oa.embedding_id
            LEFT JOIN user_comments uc ON nu.user_id = uc.user_id AND a.embedding_id = uc.embedding_id
            GROUP BY nu.user_id, a.embedding_id, a.title, a.published_date
        ),
        ratings_calculated AS (
            SELECT 
                user_id,
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
            user_id,
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

class MatrixFactorizationModel(tfrs.Model):
    """Matrix Factorization Recommender Model for News Articles"""
    
    def __init__(self, 
                 embedding_dimension: int = 64,
                 regularization: float = 0.1,
                 use_bias: bool = True,
                 user_vocab: Optional[tf.data.Dataset] = None,
                 article_vocab: Optional[tf.data.Dataset] = None):
        super().__init__()
        
        self.embedding_dimension = embedding_dimension
        self.regularization = regularization
        self.use_bias = use_bias
        
        # User and article vocabularies
        self.user_vocab = user_vocab
        self.article_vocab = article_vocab
        
        # User embedding matrix
        self.user_embedding = tf.keras.Sequential([
            tf.keras.utils.StringLookup(
                vocabulary=user_vocab, mask_token=None),
            tf.keras.layers.Embedding(
                len(user_vocab) + 1, 
                embedding_dimension,
                embeddings_regularizer=tf.keras.regularizers.l2(regularization)
            )
        ])
        
        # Article embedding matrix  
        self.article_embedding = tf.keras.Sequential([
            tf.keras.utils.StringLookup(
                vocabulary=article_vocab, mask_token=None),
            tf.keras.layers.Embedding(
                len(article_vocab) + 1, 
                embedding_dimension,
                embeddings_regularizer=tf.keras.regularizers.l2(regularization)
            )
        ])
        
        # Bias terms (optional)
        if self.use_bias:
            self.user_bias = tf.keras.Sequential([
                tf.keras.utils.StringLookup(
                    vocabulary=user_vocab, mask_token=None),
                tf.keras.layers.Embedding(len(user_vocab) + 1, 1)
            ])
            
            self.article_bias = tf.keras.Sequential([
                tf.keras.utils.StringLookup(
                    vocabulary=article_vocab, mask_token=None),
                tf.keras.layers.Embedding(len(article_vocab) + 1, 1)
            ])
            
            # Global bias
            self.global_bias = tf.Variable(0.0, trainable=True)
        
        # Rating prediction task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError()
            ]
        )
    
    def call(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Get user and article embeddings
        user_emb = self.user_embedding(features["user_id"])  # Shape: (batch_size, embedding_dim)
        article_emb = self.article_embedding(features["embedding_id"])  # Shape: (batch_size, embedding_dim)
        
        # Compute dot product for matrix factorization
        rating_prediction = tf.reduce_sum(user_emb * article_emb, axis=1, keepdims=True)
        
        # Add bias terms if enabled
        if self.use_bias:
            user_bias = self.user_bias(features["user_id"])
            article_bias = self.article_bias(features["embedding_id"])
            rating_prediction += user_bias + article_bias + self.global_bias
        
        # Clip predictions to valid rating range [0, 5]
        rating_prediction = tf.clip_by_value(rating_prediction, 0.0, 5.0)
        
        return {
            "user_embedding": user_emb,
            "article_embedding": article_emb,
            "predicted_rating": rating_prediction
        }
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        predictions = self(features)
        
        # Rating prediction loss
        rating_loss = self.rating_task(
            labels=features["rating"],
            predictions=predictions["predicted_rating"],
        )
        
        return rating_loss

class NewsMatrixFactorizationSystem:
    """Complete Matrix Factorization recommender system with training and inference"""
    
    def __init__(self, db_connection_string: str):
        self.data_pipeline = NewsFeedDataPipeline(db_connection_string)
        self.model = None
        self.train_ds = None
        self.test_ds = None
        self.user_vocab = None
        self.article_vocab = None
        
    def prepare_datasets(self, test_ratio: float = 0.2):
        """Prepare training and testing datasets"""
        # Load data
        df = self.data_pipeline.load_and_prepare_data()
        
        # Convert to tensorflow dataset format
        ratings_ds = tf.data.Dataset.from_tensor_slices({
            "user_id": df["user_id"].astype(str).values,
            "embedding_id": df["embedding_id"].astype(str).values,
            "rating": df["rating"].astype(np.float32).values,
        })
        
        # Create vocabularies
        feature_ds = ratings_ds.map(lambda x: {
            "user_id": x["user_id"],
            "embedding_id": x["embedding_id"]
        })
        
        user_ids = feature_ds.map(lambda x: x["user_id"])
        article_ids = feature_ds.map(lambda x: x["embedding_id"])
        
        self.user_vocab = user_ids.unique()
        self.article_vocab = article_ids.unique()
        
        # Shuffle and split
        shuffled = ratings_ds.shuffle(100000, seed=42, reshuffle_each_iteration=False)
        
        total_size = len(df)
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size
        
        self.train_ds = shuffled.skip(test_size).batch(8192).cache()
        self.test_ds = shuffled.take(test_size).batch(4096).cache()
        
        logger.info(f"Prepared {train_size} training and {test_size} test samples")
        logger.info(f"Number of users: {len(list(self.user_vocab))}")
        logger.info(f"Number of articles: {len(list(self.article_vocab))}")
        
    def build_model(self, embedding_dimension: int = 64, regularization: float = 0.1, use_bias: bool = True):
        """Build the matrix factorization model"""
        self.model = MatrixFactorizationModel(
            embedding_dimension=embedding_dimension,
            regularization=regularization,
            use_bias=use_bias,
            user_vocab=self.user_vocab,
            article_vocab=self.article_vocab
        )
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        logger.info(f"Built matrix factorization model with {embedding_dimension}D embeddings")
        
    def train(self, epochs: int = 10):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        logger.info("Starting model training...")
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            self.train_ds,
            epochs=epochs,
            validation_data=self.test_ds,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("Training completed!")
        return history
        
    def evaluate(self):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        metrics = self.model.evaluate(self.test_ds, return_dict=True)
        logger.info(f"Test metrics: {metrics}")
        return metrics
        
    def predict_rating(self, user_id: str, embedding_id: str):
        """Predict rating for a specific user-article pair"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Create a single sample dataset
        sample_ds = tf.data.Dataset.from_tensor_slices({
            "user_id": [user_id],
            "embedding_id": [embedding_id]
        })
        
        prediction = self.model(sample_ds.batch(1).take(1).__iter__().next())
        return prediction["predicted_rating"].numpy()[0][0]
    
    def get_user_recommendations(self, user_id: str, k: int = 10, exclude_seen: bool = True):
        """Get top-k recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get all articles
        all_articles = list(self.article_vocab.as_numpy_iterator())
        
        if exclude_seen:
            # Get articles the user has already interacted with
            df = self.data_pipeline.load_and_prepare_data()
            seen_articles = set(df[df['user_id'].astype(str) == user_id]['embedding_id'].astype(str).values)
            candidate_articles = [art.decode('utf-8') if isinstance(art, bytes) else str(art) 
                                for art in all_articles if str(art) not in seen_articles]
        else:
            candidate_articles = [art.decode('utf-8') if isinstance(art, bytes) else str(art) 
                                for art in all_articles]
        
        if not candidate_articles:
            logger.warning(f"No candidate articles found for user {user_id}")
            return []
        
        # Create dataset for all user-article pairs
        user_ids = [user_id] * len(candidate_articles)
        
        candidate_ds = tf.data.Dataset.from_tensor_slices({
            "user_id": user_ids,
            "embedding_id": candidate_articles
        }).batch(1024)
        
        # Get predictions for all candidates
        predictions = []
        for batch in candidate_ds:
            batch_predictions = self.model(batch)["predicted_rating"]
            predictions.extend(batch_predictions.numpy().flatten())
        
        # Sort and get top-k
        article_scores = list(zip(candidate_articles, predictions))
        article_scores.sort(key=lambda x: x[1], reverse=True)
        
        return article_scores[:k]
    
    def get_article_embeddings(self):
        """Get learned article embeddings"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        all_articles = list(self.article_vocab.as_numpy_iterator())
        article_ds = tf.data.Dataset.from_tensor_slices(all_articles).batch(1024)
        
        embeddings = {}
        for batch in article_ds:
            batch_embeddings = self.model.article_embedding(batch)
            for i, article in enumerate(batch.numpy()):
                article_id = article.decode('utf-8') if isinstance(article, bytes) else str(article)
                embeddings[article_id] = batch_embeddings[i].numpy()
                
        return embeddings
    
    def get_user_embeddings(self):
        """Get learned user embeddings"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        all_users = list(self.user_vocab.as_numpy_iterator())
        user_ds = tf.data.Dataset.from_tensor_slices(all_users).batch(1024)
        
        embeddings = {}
        for batch in user_ds:
            batch_embeddings = self.model.user_embedding(batch)
            for i, user in enumerate(batch.numpy()):
                user_id = user.decode('utf-8') if isinstance(user, bytes) else str(user)
                embeddings[user_id] = batch_embeddings[i].numpy()
                
        return embeddings
        
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save_weights(path)
        logger.info(f"Model weights saved to {path}")
        
    def load_model_weights(self, path: str):
        """Load model weights"""
        if self.model is None:
            raise ValueError("Build model first before loading weights.")
        self.model.load_weights(path)
        logger.info(f"Model weights loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Database configuration
    DB_CONNECTION = "postgresql://username:password@localhost:5432/newsfeed_db"
    
    # Initialize recommender system
    recommender = NewsMatrixFactorizationSystem(DB_CONNECTION)
    
    # Prepare data
    recommender.prepare_datasets(test_ratio=0.2)
    
    # Build model
    recommender.build_model(
        embedding_dimension=64, 
        regularization=0.01, 
        use_bias=True
    )
    
    # Train model
    history = recommender.train(epochs=15)
    
    # Evaluate model
    metrics = recommender.evaluate()
    print(f"Final metrics: {metrics}")
    
    # Get recommendations for a user
    try:
        recommendations = recommender.get_user_recommendations("user_123", k=10)
        print(f"Top 10 recommendations for user_123:")
        for article_id, predicted_rating in recommendations:
            print(f"  Article {article_id}: {predicted_rating:.3f}")
    except Exception as e:
        print(f"Error getting recommendations: {e}")
    
    # Predict specific rating
    try:
        rating = recommender.predict_rating("user_123", "article_456")
        print(f"Predicted rating for user_123 on article_456: {rating:.3f}")
    except Exception as e:
        print(f"Error predicting rating: {e}")
    
    # Save model
    recommender.save_model("./matrix_factorization_weights")
