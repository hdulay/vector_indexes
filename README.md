# vector_indexes


## PG Vector

Open-source vector similarity search for Postgres. Store your vectors with the rest of your data. It is implemented as a Postgres extension that needs to be installed and enabled. Follow the installation instructions for pgvector [here](https://github.com/pgvector/pgvector).

Alternatively, You can use docker compose to stand up a Postgres instance already enabled with `pgvector`.

```bash
docker run --name pgvector -d -e POSTGRES_PASSWORD=postgres ankane/pgvector
docker exec -it pgvector bash

$ psql -U postgres
```

## Sentence Transformer Model

SentenceTransformers provides models that embed images and text into the same vector space. This allows us to find similar images and implement image search. SentenceTransformers provide a wrapper for the OpenAI CLIP Model, which was trained on various (image, text)-pairs. You can read more [here](https://www.sbert.net/examples/applications/image-search/README.html?highlight=image).


## Python + Psycopg

Psycopg is a popular python driver for Postgres. We will use it directly with `pgvector` to insert and invoke similarity searches of images.

Create a Python environment and install the Python module requirements.

```bash
python -m venv .venv

pip install  --upgrade --quiet  -r requirements.txt 
```

To run the demo, execute the command below. The `-s` will seed the Postgres database with the images in the `images` directory. The `-s` can be omitted. You must reseed the database if you add images or change file names.

```bash
$ python example-psycopg.py -s

seeding the database
Enter image query: show me a picture of NYC
```

The application will pop up a picture closely matching your query and the distance value.

## Vecs

Similar to LangChain, Vecs is a Python client library for managing and querying vector stores in PostgreSQL, leveraging the capabilities of the pgvector extension.

```sql
\dt vecs.image_vectors
```

