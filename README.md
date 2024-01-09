# vector_indexes



## Docker

```bash
docker run --name pgvector -d -e POSTGRES_PASSWORD=postgres ankane/pgvector
docker exec -it pgvector bash
```

## Vector Setup

```sql
CREATE EXTENSION vector;
```

## Python Setup




