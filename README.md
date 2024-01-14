# vector_indexes


## PG Vector

Open-source vector similarity search for Postgres. Store your vectors with the rest of your data.

## Sentence Transformers

SentenceTransformers provides models that allow to embed images and text into the same vector space. This allows to find similar images as well as to implement image search. SentenceTransformers provides a wrapper for the OpenAI CLIP Model, which was trained on a variety of (image, text)-pairs. You can read more [here](https://www.sbert.net/examples/applications/image-search/README.html?highlight=image).

## Vecs

Similar to LangChain, Vecs is a Python client library for managing and querying vector stores in PostgreSQL, leveraging the capabilities of the pgvector extension.

## OpenAI

Create a `.env` file to add your OpenAPI key

```properties
OPENAI_API_KEY=foobar
```

## Python

```bash
pip install  --upgrade --quiet  -r requirements.txt 
```

## Postgres

```bash
docker run --name pgvector -d -e POSTGRES_PASSWORD=postgres ankane/pgvector
docker exec -it pgvector bash

$ psql -U postgres

```

```sql
CREATE TABLE pics (id bigserial PRIMARY KEY, embedding vector(512));
```

## Vector Index Setup

```sql
CREATE EXTENSION vector;
```

## Vecs + Postgres

```sql
\dt vecs.image_vectors
```


