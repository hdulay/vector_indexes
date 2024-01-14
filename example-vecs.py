from PIL import Image
from sentence_transformers import SentenceTransformer
import vecs
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os, sys

DB_CONNECTION = "postgresql://postgres:postgres@localhost:5432/postgres"

def seed():
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)

    # create a collection of vectors with 3 dimensions
    images = vx.get_or_create_collection(name="image_vectors", dimension=512)

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')

    # Encode an image:
    images = os.listdir("./images")
    records = []
    for f in images:
        file = f'./images/{f}'
        img_emb = model.encode(Image.open(file))
        records.append((file, img_emb, {"type": "jpg"}))

    # add records to the *images* collection
    images.upsert(records=records)
    print("Inserted images")

    # index the collection for fast search performance
    images.create_index()
    print("Created index")

def search():
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)
    images = vx.get_or_create_collection(name="image_vectors", dimension=512)

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')
    # Encode text query
    query_string = input("Enter image query:")
    # query_string = "a picture of black grapes on a vine"
    text_emb = model.encode(query_string)

    # query the collection filtering metadata for "type" = "jpg"
    results = images.query(
        data=text_emb,                      # required
        limit=5,                            # number of records to return
        filters={"type": {"$eq": "jpg"}},   # metadata filters
        include_value=True,
    )
    result = results[0]
    print(results)
    file = result[0]
    plt.title(file)
    image = mpimg.imread('./images/' + file)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1 and args[1] == '-s':
        seed()

    search()
