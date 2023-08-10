import pinecone
import torch
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

with open('secrets.json', 'r') as f:
    secrets = json.load(f)

### Define Global Variables
PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']

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
model = LanguageModel(
    MODEL_PATH, generation_config=generation_config,
    device_map='auto', load_in_8bit=True,
    system_tag=SYSTEM_TAG, user_tag=USER_TAG, ai_tag=AI_TAG
)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

prompter = EPS(index, embedding_model)
agent = PWS(model)

while(true):
    task = input("Please enter your question")
    selection = prompter.select_examples(task)
    examples = [entry['metadata'] for entry in selection]
    response = agent.run(question, examples)
    print(response['output'])
    print(f"Question answered in {response['wall_time']} seconds.")
