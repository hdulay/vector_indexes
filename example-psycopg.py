
from pgvector.psycopg2 import register_vector
import psycopg
from sentence_transformers import SentenceTransformer
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from pgvector.psycopg import register_vector

conn = psycopg.connect(dbname="postgres", autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)
conn.execute('DROP TABLE IF EXISTS items')
conn.execute('CREATE TABLE items (id bigserial PRIMARY KEY, path varchar(64), embedding vector(512))')


# CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(512));
# ALTER TABLE items ADD COLUMN embedding vector(512);

model = SentenceTransformer('clip-ViT-B-32')
img_emb1 = model.encode(Image.open('./images/one.jpg'))
img_emb2 = model.encode(Image.open('./images/two.jpg'))
img_emb3 = model.encode(Image.open('./images/three.jpg'))
img_emb4 = model.encode(Image.open('./images/four.jpg'))

cur = conn.cursor()
cur.execute('create extension if not exists vector with schema public')
cur.execute('INSERT INTO items (embedding, path) VALUES (%s,%s)', (img_emb1.tolist(),'./images/one.jpg'))
cur.execute('INSERT INTO items (embedding, path) VALUES (%s,%s)', (img_emb2.tolist(),'./images/two.jpg'))
cur.execute('INSERT INTO items (embedding, path) VALUES (%s,%s)', (img_emb3.tolist(),'./images/three.jpg'))
cur.execute('INSERT INTO items (embedding, path) VALUES (%s,%s)', (img_emb4.tolist(),'./images/four.jpg'))

# query_string = "a white bike in front of a red brick wall"
query_string = input("Enter image query:")
text_emb = model.encode(query_string)

cur.execute("""
            SELECT id, path, embedding <-> %s AS distance 
            FROM items ORDER BY embedding::vector(512) <-> %s LIMIT 4
            """, 
            (str(text_emb.tolist()),str(text_emb.tolist())))

rows = cur.fetchall()
print(rows[0])
plt.title(rows[0][1])
image = mpimg.imread(rows[0][1])
plt.imshow(image)
plt.show()


