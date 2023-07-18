import os
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

from utils import LanguageModel, EPS, PWS

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT')
MODEL_PATH = "lmsys/vicuna-7b-v1.3"
TEMPERATURE = 0.7
TOP_K = 100
TOP_P = 0.7
REPETITION_PENALTY= 1.0
MAX_NEW_TOKENS = 512

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'plans'
index = pinecone.GRPCIndex(index_name)

generation_config = GenerationConfig(
    do_sample=True,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    max_new_tokens=MAX_NEW_TOKENS
)

model = LanguageModel(MODEL_PATH, generation_config=generation_config)
prompter = EPS(index, embedding_model)
agent = PWS(model)

while(True):
    task = input("Enter your question...")
    selection = prompter.select_examples(task)
    examples = [entry['metadata'] for entry in selection]
    print(agent.run(task, examples))


