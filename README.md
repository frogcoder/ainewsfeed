# AI News Feed
## Running
`init.sql` is the script to initialize the database in Postgresql.

if you are running `docker compose up` please create directory `./initdb/` and copy init.sql into the folder.

## Environment Variables
### Required
- TF_USE_LEGACY_KERAS="1"

### Optional
- CHROMA_PATH
  The path of directory that stores ChromaDB data.  Default: `./chroma_articles_db`
- POSTGRES_URL
  The connection url to the database.  Default: `postgresql://newsfeed:newsfeed@postgres:5432/newsfeed`

### Initialize the data
When running for the first time the following actions must be invoked in the specified order.  You can execute these in the `/docs` url, for example `http://localhost:8000/docs`.

1. Scrap
   `POST /scrape`
2. Add at least one user
   `POST /users`
   ```json
   {
       "user_id": "newuser"
   }
   ```
3. Train the model
   `POST /train`
