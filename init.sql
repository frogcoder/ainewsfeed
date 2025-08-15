CREATE SEQUENCE IF NOT EXISTS public.articles_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

CREATE SEQUENCE IF NOT EXISTS public.newsfeed_users_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

CREATE SEQUENCE IF NOT EXISTS public.opened_articles_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

CREATE SEQUENCE IF NOT EXISTS public.user_comments_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;


CREATE TABLE IF NOT EXISTS public.newsfeed_users
(
    id integer NOT NULL DEFAULT nextval('newsfeed_users_id_seq'::regclass),
    user_id text COLLATE pg_catalog."default",
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT newsfeed_users_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.articles
(
    id integer NOT NULL DEFAULT nextval('articles_id_seq'::regclass),
    title text COLLATE pg_catalog."default" NOT NULL,
    content text COLLATE pg_catalog."default" NOT NULL,
    url text COLLATE pg_catalog."default" NOT NULL,
    source text COLLATE pg_catalog."default" NOT NULL,
    published_date timestamp without time zone,
    author text COLLATE pg_catalog."default",
    summary text COLLATE pg_catalog."default",
    tags text[] COLLATE pg_catalog."default",
    content_hash text COLLATE pg_catalog."default" NOT NULL,
    scraped_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    embedding_id text COLLATE pg_catalog."default",
    CONSTRAINT articles_pkey PRIMARY KEY (id),
    CONSTRAINT articles_content_hash_key UNIQUE (content_hash),
    CONSTRAINT articles_url_key UNIQUE (url)
);

CREATE INDEX IF NOT EXISTS idx_articles_content_hash
    ON public.articles USING btree
    (content_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_articles_published_date

-- DROP INDEX IF EXISTS public.idx_articles_published_date;

CREATE INDEX IF NOT EXISTS idx_articles_published_date
    ON public.articles USING btree
    (published_date ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_articles_source

-- DROP INDEX IF EXISTS public.idx_articles_source;

CREATE INDEX IF NOT EXISTS idx_articles_source
    ON public.articles USING btree
    (source COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;


CREATE TABLE IF NOT EXISTS public.opened_articles
(
    id integer NOT NULL DEFAULT nextval('opened_articles_id_seq'::regclass),
    embedding_id text COLLATE pg_catalog."default" NOT NULL,
    user_id text COLLATE pg_catalog."default" NOT NULL,
    opened_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT opened_articles_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.user_comments
(
    id integer NOT NULL DEFAULT nextval('user_comments_id_seq'::regclass),
    user_id text COLLATE pg_catalog."default" NOT NULL,
    embedding_id text COLLATE pg_catalog."default" NOT NULL,
    comment text COLLATE pg_catalog."default" NOT NULL,
    is_public boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_comments_pkey PRIMARY KEY (id),
    CONSTRAINT chk_comment_not_empty CHECK (length(TRIM(BOTH FROM comment)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_user_comments_created_at
    ON public.user_comments USING btree
    (created_at ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_user_comments_embedding_id

-- DROP INDEX IF EXISTS public.idx_user_comments_embedding_id;

CREATE INDEX IF NOT EXISTS idx_user_comments_embedding_id
    ON public.user_comments USING btree
    (embedding_id COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_user_comments_is_public

-- DROP INDEX IF EXISTS public.idx_user_comments_is_public;

CREATE INDEX IF NOT EXISTS idx_user_comments_is_public
    ON public.user_comments USING btree
    (is_public ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_user_comments_user_embedding

-- DROP INDEX IF EXISTS public.idx_user_comments_user_embedding;

CREATE INDEX IF NOT EXISTS idx_user_comments_user_embedding
    ON public.user_comments USING btree
    (user_id COLLATE pg_catalog."default" ASC NULLS LAST, embedding_id COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_user_comments_user_id

-- DROP INDEX IF EXISTS public.idx_user_comments_user_id;

CREATE INDEX IF NOT EXISTS idx_user_comments_user_id
    ON public.user_comments USING btree
    (user_id COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;


CREATE OR REPLACE FUNCTION public.update_updated_at_column()
    RETURNS trigger
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE NOT LEAKPROOF
AS $BODY$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$BODY$;

-- Trigger: update_user_comments_updated_at

-- DROP TRIGGER IF EXISTS update_user_comments_updated_at ON public.user_comments;

CREATE OR REPLACE TRIGGER update_user_comments_updated_at
    BEFORE UPDATE 
    ON public.user_comments
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();
