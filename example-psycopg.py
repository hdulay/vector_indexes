import psycopg
from sentence_transformers import SentenceTransformer
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from pgvector.psycopg import register_vector
import os, sys


conn = psycopg.connect(dbname="postgres", autocommit=True)
model = SentenceTransformer('clip-ViT-B-32')

def seed():
    print("seeding the database")
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    conn.execute('DROP TABLE IF EXISTS items')
    conn.execute('CREATE TABLE items (id bigserial PRIMARY KEY, path varchar(64), embedding vector(512))')
    cur = conn.cursor()
    cur.execute('create extension if not exists vector with schema public')

    images = os.listdir("./images")
    for f in images:
        file = f'./images/{f}'
        img_emb = model.encode(Image.open(file))
        cur.execute('INSERT INTO items (embedding, path) VALUES (%s,%s)', (img_emb.tolist(), file))

def search():    
    # query_string = "a white bike in front of a red brick wall"
    query_string = input("Enter image query:")
    text_emb = model.encode(query_string)

    cur = conn.cursor()
    cur.execute("""
                SELECT id, path, embedding <-> %s AS distance 
                FROM items ORDER BY embedding::vector(512) <-> %s
                """, 
                (str(text_emb.tolist()),str(text_emb.tolist())))

    rows = cur.fetchall()
    print(rows)
    show(rows[0][1], rows[0][2])
    show(rows[1][1], rows[0][2])
    
def show(path, distance):
    plt.title(f'{path} {distance}')
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':

    args = sys.argv

    if len(args) > 1 and args[1] == '-s':
        seed()

    search()

