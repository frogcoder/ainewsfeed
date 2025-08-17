import asyncio
import os
from typing import Annotated
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import newsscrapper
import recommender
import db


class User(BaseModel):
    user_id: str


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

model = None


@app.post("/scrape")
async def scrape():
    await newsscrapper.scrape()


@app.post("/train")
def train():
    global model
    model = recommender.train_model()


@app.get("/articles/{user_id}")
async def get_recommendations(user_id: str):
    print("querying articles for", user_id)
    embedding_ids = recommender.get_recommendations(model, user_id, 20)
    print("found", len(embedding_ids), "articles")
    articles = db.query_articles(embedding_ids)
    print(articles)
    return articles


@app.get("/related/{embedding_id}")
def get_related_articles(embedding_id: str):
    related_articles = db.search_related_articles(embedding_id)
    return related_articles


@app.get("/", response_class=HTMLResponse)
def newsfeed(request: Request):
    return templates.TemplateResponse(
        request=request, name="newsfeed.html", context={"userId": "seanliu"}
    )


@app.post("/", response_class=HTMLResponse)
def newsfeed(request: Request, user_id: Annotated[str, Form()]):
    print("user", user_id, "logging in")
    return templates.TemplateResponse(
        request=request, name="newsfeed.html", context={"userId": user_id}
    )


@app.get("/login", response_class=HTMLResponse)
def login(request: Request, response_class=HTMLResponse):
    return templates.TemplateResponse(
        request=request, name="login.html"
    )

@app.post("/users/")
def create_user(user: User):
    db.create_user(user.user_id)
