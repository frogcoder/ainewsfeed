from typing import Dict, Text, Optional
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import datasets


ratings_ds = datasets.load_ratings()
articles_ds = datasets.load_articles()

ratings = ratings_ds.map(lambda x: {
    "embedding_id": x["embedding_id"],
    "user_id": x["user_id"],
    "rating": x["rating"]
})
articles = articles_ds.map(lambda x: x["embedding_id"])

shuffled = ratings.shuffle(100_000, reshuffle_each_iteration=False)

train_size = round(len(ratings) * 0.8)
train = shuffled.take(train_size)
test = shuffled.skip(train_size)

article_ids = articles.batch(100)
user_ids = ratings.batch(100).map(lambda x: x["user_id"])

unique_article_ids = np.unique(np.concatenate(list(article_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))



#The full model
class FeedRecommenderModel(tfrs.Model):
    def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        super().__init__()
        embedding_dimension = 32

        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        article_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_article_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_article_ids) + 1, embedding_dimension)
        ])

        self.user_model: tf.keras.layers.Layer = user_model
        self.article_model: tf.keras.layers.Layer = article_model

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        retrieval_metrics = tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(article_model)
        )
        
        retrieval_task = tfrs.tasks.Retrieval(metrics=retrieval_metrics)

        self.rating_task: tf.keras.layers.Layer = rating_task
        self.retrieval_task: tf.keras.layers.Layer = retrieval_task

        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        article_embeddings = self.article_model(features["embedding_id"])
        return (
            user_embeddings,
            article_embeddings,
            self.rating_model(
                tf.concat([user_embeddings, article_embeddings], axis=1)
            )
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")

        user_embeddings, article_embeddings, rating_predictions = self(features)

        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions
        )
        retrieval_loss = self.retrieval_task(user_embeddings, article_embeddings)
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)


def train_recommender() -> FeedRecommenderModel:
    model = FeedRecommenderModel(rating_weight=1.0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(128).cache()
    cached_test = test.batch(64).cache()

    model.fit(cached_train, epochs=3)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")


def get_rated_recommendations(model: FeedRecommenderModel, user_id: str, max_count: int):
    #Making predictions
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((articles.batch(100), articles.batch(100).map(model.article_model)))
    )

    #Get recommendations
    _, embeddings = index(tf.constant(["seanliu"]))
    print("Recommendation for user seanliu:", embeddings[0, :3])

    user_article = [
        {"user_id": np.array(["seanliu"]), "embedding_id": np.array([str(embedding_id)])}
        for embedding_id in embeddings[0, :10]
    ]

    embedding_ids = []
    for p in user_article:
        trained_movie_embeddings, trained_user_embeddings, predicted_rating = model(p)
        yield (predicted_rating, p)

    
def get_recommendations(model: FeedRecommenderModel, user_id: str, max_count: str) -> list[str]:
    rated_embeddings = get_rated_recommendations(model, user_id, max_count)
    return [embedding_id for _, embedding_id in sorted(rated_embeddings, reverse=True)]

