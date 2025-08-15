"""
Comprehensive Article Scraper with ChromaDB and PostgreSQL Storage
Scrapes articles from top news websites, Hacker News, and Reddit
"""

import asyncio
import aiohttp
import asyncpg
import chromadb
from chromadb.config import Settings
import requests
from bs4 import BeautifulSoup

import feedparser
import json
from datetime import datetime, timedelta
import hashlib
import re
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import os
from newspaper import Article
import nltk

POSTGRES_URL = "postgresql://newsfeed:newsfeed@postgres:5432/newsfeed"
CHROMA_PATH = "./chroma_articles_db"

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


@dataclass
class ArticleData:
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[datetime]
    author: Optional[str]
    summary: Optional[str]
    tags: List[str]
    content_hash: str

class DatabaseManager:
    def __init__(self, postgres_url: str, chroma_path: str = "./chroma_db"):
        self.postgres_url = postgres_url
        self.chroma_path = chroma_path
        self.pool = None
        self.chroma_client = None
        self.collection = None
        
    async def init_postgres(self):
        """Initialize PostgreSQL connection pool and create tables"""
        self.pool = await asyncpg.create_pool(self.postgres_url)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    published_date TIMESTAMP,
                    author TEXT,
                    summary TEXT,
                    tags TEXT[],
                    content_hash TEXT UNIQUE NOT NULL,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
                CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles(published_date);
                CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
            """)
    
    def init_chroma(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="articles",
            metadata={"description": "News articles and long-form content"}
        )

    async def query_articles(self, embedding_ids):
        async with self.pool.acquire() as conn:
            articles = await conn.fetch("SELECT * FROM articles WHERE embedding_id IN $1", embedding_ids)
            return articles

        
    async def store_article(self, article: ArticleData) -> bool:
        """Store article in both PostgreSQL and ChromaDB"""
        try:
            # Store in PostgreSQL
            async with self.pool.acquire() as conn:
                article_id = await conn.fetchval("""
                    INSERT INTO articles 
                    (title, content, url, source, published_date, author, summary, tags, content_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (content_hash) DO NOTHING
                    RETURNING id;
                """, article.title, article.content, article.url, article.source,
                    article.published_date, article.author, article.summary, 
                    article.tags, article.content_hash)
                
                if not article_id:
                    return False  # Already exists
            
            # Store in ChromaDB
            embedding_id = f"{article.source}_{article.content_hash[:16]}"
            self.collection.add(
                documents=[article.content],
                metadatas=[{
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "author": article.author or "",
                    "summary": article.summary or "",
                    "tags": ",".join(article.tags)
                }],
                ids=[embedding_id]
            )
            
            # Update PostgreSQL with embedding ID
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE articles SET embedding_id = $1 WHERE content_hash = $2",
                    embedding_id, article.content_hash
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing article {article.url}: {e}")
            return False


class NewsSourceScraper:
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Top 50 news and long-form content websites
        self.news_sources = {
            'nytimes.com': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
            'washingtonpost.com': 'https://feeds.washingtonpost.com/rss/world',
            'wsj.com': 'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
            'theguardian.com': 'https://www.theguardian.com/world/rss',
            'reuters.com': 'https://feeds.reuters.com/reuters/topNews',
            'bbc.com': 'https://feeds.bbci.co.uk/news/rss.xml',
            'cnn.com': 'http://rss.cnn.com/rss/edition.rss',
            'npr.org': 'https://feeds.npr.org/1001/rss.xml',
            'apnews.com': 'https://feeds.apnews.com/urn:newsml:ap.org:20120918:topnews',
            'bloomberg.com': 'https://feeds.bloomberg.com/markets/news.rss',
            'ft.com': 'https://www.ft.com/rss/home',
            'economist.com': 'https://www.economist.com/rss.xml',
            'newyorker.com': 'https://www.newyorker.com/feed/everything',
            'theatlantic.com': 'https://www.theatlantic.com/feed/all/',
            'wired.com': 'https://www.wired.com/feed/rss',
            'techcrunch.com': 'https://techcrunch.com/feed/',
            'arstechnica.com': 'https://feeds.arstechnica.com/arstechnica/index',
            'theverge.com': 'https://www.theverge.com/rss/index.xml',
            'engadget.com': 'https://www.engadget.com/rss.xml',
            'mashable.com': 'https://feeds.mashable.com/Mashable',
            'medium.com': 'https://medium.com/feed',
            'substack.com': None,  # Requires individual publication feeds
            'politico.com': 'https://www.politico.com/rss/politicopicks.xml',
            'axios.com': 'https://api.axios.com/feed/',
            'vox.com': 'https://www.vox.com/rss/index.xml',
            'buzzfeed.com': 'https://www.buzzfeed.com/index.xml',
            'huffpost.com': 'https://www.huffpost.com/section/front-page/feed',
            'usatoday.com': 'https://rssfeeds.usatoday.com/usatoday-NewsTopStories',
            'latimes.com': 'https://www.latimes.com/rss2.0.xml',
            'chicagotribune.com': 'https://www.chicagotribune.com/arcio/rss/',
            'nbcnews.com': 'https://feeds.nbcnews.com/nbcnews/public/news',
            'abcnews.go.com': 'https://abcnews.go.com/abcnews/topstories',
            'cbsnews.com': 'https://www.cbsnews.com/latest/rss/main',
            'foxnews.com': 'https://feeds.foxnews.com/foxnews/latest',
            'time.com': 'https://feeds2.feedburner.com/time/topstories',
            'newsweek.com': 'https://www.newsweek.com/rss',
            'slate.com': 'https://slate.com/feeds/all.rss',
            'salon.com': 'https://www.salon.com/feed/',
            'dailybeast.com': 'https://feeds.thedailybeast.com/rss/articles',
            'propublica.org': 'https://feeds.propublica.org/propublica/main',
            'vice.com': 'https://www.vice.com/en/rss',
            'motherboard.vice.com': 'https://www.vice.com/en/rss/topic/tech',
            'scientificamerican.com': 'https://rss.sciam.com/ScientificAmerican-Global',
            'nature.com': 'https://www.nature.com/nature.rss',
            'nationalgeographic.com': 'https://www.nationalgeographic.com/pages/feed.rss',
            'smithsonianmag.com': 'https://www.smithsonianmag.com/rss/latest_articles/',
            'theintercept.com': 'https://theintercept.com/feed/?rss',
            'foreignaffairs.com': 'https://www.foreignaffairs.com/rss.xml',
            'foreignpolicy.com': 'https://foreignpolicy.com/feed',
            'stratfor.com': 'https://worldview.stratfor.com/rss.xml',
            'longform.org': 'https://longform.org/feed',
            'longreads.com': 'https://longreads.com/rss/'
        }
    
    async def init_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def scrape_rss_feed(self, source: str, rss_url: str, limit: int = 10) -> List[ArticleData]:
        """Scrape articles from RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                try:
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    article.nlp()
                    
                    # Create content hash
                    content_hash = hashlib.md5(article.text.encode()).hexdigest()
                    
                    # Parse published date
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    
                    article_data = ArticleData(
                        title=article.title or entry.title,
                        content=article.text,
                        url=entry.link,
                        source=source,
                        published_date=published_date,
                        author=", ".join(article.authors) if article.authors else None,
                        summary=article.summary,
                        tags=article.keywords if article.keywords else [],
                        content_hash=content_hash
                    )
                    
                    if len(article_data.content) > 500:  # Only store substantial articles
                        articles.append(article_data)
                        
                except Exception as e:
                    logging.error(f"Error processing article {entry.link}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error scraping RSS feed {rss_url}: {e}")
        
        return articles
    
    async def scrape_website_direct(self, source: str, url: str, limit: int = 10) -> List[ArticleData]:
        """Directly scrape website for articles"""
        articles = []
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find article links (common patterns)
                    article_links = set()
                    for selector in ['a[href*="/article"]', 'a[href*="/story"]', 'a[href*="/news"]', 
                                   'a[href*="/post"]', '.article-link', '.story-link']:
                        links = soup.select(selector)
                        for link in links[:limit]:
                            href = link.get('href')
                            if href:
                                full_url = urljoin(url, href)
                                article_links.add(full_url)
                    
                    # Process found articles
                    for article_url in list(article_links)[:limit]:
                        try:
                            article = Article(article_url)
                            article.download()
                            article.parse()
                            article.nlp()
                            
                            if len(article.text) > 500:
                                content_hash = hashlib.md5(article.text.encode()).hexdigest()
                                
                                article_data = ArticleData(
                                    title=article.title,
                                    content=article.text,
                                    url=article_url,
                                    source=source,
                                    published_date=article.publish_date,
                                    author=", ".join(article.authors) if article.authors else None,
                                    summary=article.summary,
                                    tags=article.keywords if article.keywords else [],
                                    content_hash=content_hash
                                )
                                articles.append(article_data)
                                
                        except Exception as e:
                            logging.error(f"Error processing direct article {article_url}: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"Error scraping website {url}: {e}")
        
        return articles

class HackerNewsScraper:
    def __init__(self):
        self.base_url = "https://hacker-news.firebaseio.com/v0"
    
    async def get_top_stories(self, limit: int = 50) -> List[ArticleData]:
        """Scrape top stories from Hacker News"""
        articles = []
        
        try:
            # Get top story IDs
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/topstories.json") as response:
                    story_ids = await response.json()
                
                # Process stories
                for story_id in story_ids[:limit]:
                    try:
                        async with session.get(f"{self.base_url}/item/{story_id}.json") as response:
                            story = await response.json()
                        
                        if story.get('type') == 'story' and story.get('url'):
                            # Try to get full article content
                            try:
                                article = Article(story['url'])
                                article.download()
                                article.parse()
                                
                                if len(article.text) > 300:
                                    content = article.text
                                else:
                                    content = story.get('text', story['title'])
                                
                                content_hash = hashlib.md5(content.encode()).hexdigest()
                                published_date = datetime.fromtimestamp(story['time']) if story.get('time') else None
                                
                                article_data = ArticleData(
                                    title=story['title'],
                                    content=content,
                                    url=story['url'],
                                    source='hackernews',
                                    published_date=published_date,
                                    author=story.get('by'),
                                    summary=story['title'],  # Use title as summary for HN
                                    tags=['hackernews', 'tech'],
                                    content_hash=content_hash
                                )
                                articles.append(article_data)
                                
                            except Exception as e:
                                # Fallback to HN text if article scraping fails
                                content = story.get('text', story['title'])
                                if len(content) > 50:
                                    content_hash = hashlib.md5(content.encode()).hexdigest()
                                    published_date = datetime.fromtimestamp(story['time']) if story.get('time') else None
                                    
                                    article_data = ArticleData(
                                        title=story['title'],
                                        content=content,
                                        url=story['url'],
                                        source='hackernews',
                                        published_date=published_date,
                                        author=story.get('by'),
                                        summary=story['title'],
                                        tags=['hackernews', 'tech'],
                                        content_hash=content_hash
                                    )
                                    articles.append(article_data)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logging.error(f"Error processing HN story {story_id}: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error scraping Hacker News: {e}")
        
        return articles



class ArticleScrapingSystem:
    def __init__(self, postgres_url: str, chroma_path: str = "./chroma_db"):
        self.db_manager = DatabaseManager(postgres_url, chroma_path)
        self.news_scraper = NewsSourceScraper()
        self.hn_scraper = HackerNewsScraper()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
    
    async def initialize(self):
        """Initialize all components"""
        await self.db_manager.init_postgres()
        self.db_manager.init_chroma()
        await self.news_scraper.init_session()
        logging.info("System initialized successfully")
    
    async def scrape_all_sources(self, articles_per_source: int = 10):
        """Scrape articles from all sources"""
        all_articles = []
        
        # Scrape news sources
        logging.info("Starting news source scraping...")
        for source, rss_url in self.news_scraper.news_sources.items():
            if rss_url:
                try:
                    articles = await self.news_scraper.scrape_rss_feed(source, rss_url, articles_per_source)
                    all_articles.extend(articles)
                    logging.info(f"Scraped {len(articles)} articles from {source}")
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logging.error(f"Error scraping {source}: {e}")
        
        # Scrape Hacker News
        logging.info("Starting Hacker News scraping...")
        try:
            hn_articles = await self.hn_scraper.get_top_stories(articles_per_source * 2)
            all_articles.extend(hn_articles)
            logging.info(f"Scraped {len(hn_articles)} articles from Hacker News")
        except Exception as e:
            logging.error(f"Error scraping Hacker News: {e}")
        
        return all_articles
    
    async def store_articles(self, articles: List[ArticleData]):
        """Store all articles in databases"""
        stored_count = 0
        duplicate_count = 0
        
        for article in articles:
            try:
                success = await self.db_manager.store_article(article)
                if success:
                    stored_count += 1
                    logging.info(f"Stored: {article.title[:50]}...")
                else:
                    duplicate_count += 1
                    
            except Exception as e:
                logging.error(f"Error storing article {article.url}: {e}")
        
        logging.info(f"Storage complete: {stored_count} new articles, {duplicate_count} duplicates")
        return stored_count, duplicate_count
    
    async def run_scraping_cycle(self, articles_per_source: int = 10):
        """Run a complete scraping cycle"""
        start_time = datetime.now()
        logging.info("Starting scraping cycle...")
        
        try:
            # Scrape all sources
            articles = await self.scrape_all_sources(articles_per_source)
            logging.info(f"Total articles scraped: {len(articles)}")
            
            # Store articles
            stored, duplicates = await self.store_articles(articles)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logging.info(f"Scraping cycle completed in {duration}")
            logging.info(f"Results: {stored} new articles stored, {duplicates} duplicates skipped")
            
            return {
                'total_scraped': len(articles),
                'new_stored': stored,
                'duplicates': duplicates,
                'duration': duration
            }
            
        except Exception as e:
            logging.error(f"Error in scraping cycle: {e}")
            raise
    
    async def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """Search articles in ChromaDB"""
        try:
            results = self.db_manager.collection.query(
                query_texts=[query],
                n_results=limit
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
            
            return search_results
            
        except Exception as e:
            logging.error(f"Error searching articles: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.news_scraper.close_session()
        if self.db_manager.pool:
            await self.db_manager.pool.close()
        logging.info("Cleanup completed")

# Usage example and configuration
async def main():
    # Configuration
    
    # Initialize system
    scraper_system = ArticleScrapingSystem(
        postgres_url=POSTGRES_URL,
        chroma_path=CHROMA_PATH
    )
    
    try:
        await scraper_system.initialize()
        
        # Run scraping cycle
        #results = await scraper_system.run_scraping_cycle(articles_per_source=15)
        #print(f"Scraping completed: {results}")
        
        # Example search
        search_results = await scraper_system.search_articles("artificial intelligence", limit=5)
        print(f"Found {len(search_results)} relevant articles")
        for r in search_results:
            print(r)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        await scraper_system.cleanup()

# Scheduled scraping function
async def run_scheduled_scraping():
    """Run scraping on a schedule"""
    scraper_system = ArticleScrapingSystem(
        postgres_url=os.getenv("POSTGRES_URL", POSTGRES_URL),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_articles_db")
    )
    
    try:
        await scraper_system.initialize()
        
        while True:
            try:
                logging.info("Starting scheduled scraping cycle...")
                results = await scraper_system.run_scraping_cycle(articles_per_source=20)
                logging.info(f"Scheduled cycle completed: {results}")
                
                # Wait 4 hours before next cycle
                await asyncio.sleep(4 * 60 * 60)
                
            except Exception as e:
                logging.error(f"Error in scheduled cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
                
    except KeyboardInterrupt:
        logging.info("Scheduled scraping stopped by user")
    finally:
        await scraper_system.cleanup()

if __name__ == "__main__":
    # For one-time scraping
    asyncio.run(main())
    
    # For scheduled scraping (uncomment to use)
    # asyncio.run(run_scheduled_scraping())

# Requirements.txt content:
"""
aiohttp>=3.8.0
asyncpg>=0.27.0
chromadb>=0.4.0
requests>=2.31.0
beautifulsoup4>=4.12.0
feedparser>=6.0.0
newspaper3k>=0.2.8
nltk>=3.8
psycopg2-binary>=2.9.0
python-dateutil>=2.8.0
"""

# Environment variables needed:
"""
POSTGRES_URL=postgresql://username:password@localhost:5432/articles_db
"""
