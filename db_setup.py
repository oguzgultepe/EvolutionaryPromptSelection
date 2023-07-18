import os
import re

import pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

regex_pattern = r'(.*)\[input\]'
regex = re.compile(regex_pattern)

dataset_name = "rewoo/planner_instruction_tuning_2k"
dataset = load_dataset(dataset_name)
seed_data = dataset['train']

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'plans'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric='cosine'
    )

index = pinecone.GRPCIndex(index_name)

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
                          'dataset_name': dataset_name,
                          'score': 1,
                          'id': x})

    embeddings = embedding_model.encode(seed_data[i:i_end]['input'])
    records = zip(ids, embeddings, metadatas)
    index.upsert(vectors=records)

