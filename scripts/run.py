import json
import os
import sys

import pinecone
import torch

from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

# Add EPS dir to sys.path to import functions from EPS/src/
# Looks ugly and breaks PEP8 (E402)
# Unfortunately, there seems to be no better alternative
EPS_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(os.path.dirname(EPS_DIR))

from EPS.src.nodes import PWS  # noqa: E402
from EPS.src.utils import LMWrapper, EPS  # noqa: E402

with open('../secrets.json', 'r') as f:
    secrets = json.load(f)

# Define Global Variables
PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
INDEX_NAME = 'plans'

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

MODEL_PATH = "stabilityai/StableBeluga-13B"
SYSTEM_TAG = "### System:\n"
USER_TAG = "### User:\n"
AI_TAG = "### Assistant:\n"

# Initialize database connection
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index = pinecone.GRPCIndex(INDEX_NAME)

# Initialize models
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.01,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=256
)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
)
lm_wrapper = LMWrapper(
    model, tokenizer, generation_config,
    system_tag=SYSTEM_TAG, user_tag=USER_TAG, ai_tag=AI_TAG
)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

prompter = EPS(index, embedding_model)
agent = PWS(lm_wrapper)

while True:
    task = input("Please enter your question")
    selection = prompter.select_examples(task)
    examples = [entry['metadata'] for entry in selection]
    response = agent.run(task, examples)
    print(response['output'])
    print(f"Question answered in {response['wall_time']} seconds.")
