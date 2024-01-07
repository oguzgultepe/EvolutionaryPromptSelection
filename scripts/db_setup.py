import json
import re
import pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

with open('../secrets.json', 'r') as f:
    secrets = json.load(f)

PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
INDEX_NAME = 'plans'

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

REGEX_PATTERN = r'(.*)\[input\]'
DATASET_NAME = "rewoo/planner_instruction_tuning_2k"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

regex = re.compile(REGEX_PATTERN)
dataset = load_dataset(DATASET_NAME)
seed_data = dataset['train']

# only create index if it doesn't exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric='cosine'
    )

index = pinecone.GRPCIndex(INDEX_NAME)

size = seed_data.num_rows
batch_size = 128

for i in tqdm(range(0, size, batch_size)):
    # find end of batch
    i_end = min(i + batch_size, size)
    ids = []
    metadatas = []
    for x in range(i, i_end):
        # create IDs
        ids.append(str(x))
        # create metadata
        instance = seed_data[x]
        metadatas.append({'question': instance['input'],
                          'plan': instance['output'],
                          'tools': regex.findall(instance['instruction']),
                          'dataset_name': DATASET_NAME,
                          'score': 1,
                          'id': x})

    embeddings = embedding_model.encode(seed_data[i:i_end]['input'])
    records = zip(ids, embeddings, metadatas)
    index.upsert(vectors=records)
