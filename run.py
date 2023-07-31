import json
import string
import pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig
from tqdm.auto import tqdm

from utils import LanguageModel, EPS, PWS

with open('secrets.json', 'r') as f:
    secrets = json.load(f)

PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
INDEX_NAME = 'plans'

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

MODEL_PATH = "NousResearch/Nous-Hermes-Llama2-13b"
SYSTEM_TAG = "### Instruction:\n"
USER_TAG = "### Input:\n"
AI_TAG = "### Response:\n"

LOAD_IN_8BIT = True
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

TEMPERATURE = 0.5
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY= 1.0
MAX_NEW_TOKENS = 256

DATASET_NAME = "trivia_qa"

SIMILAR_POOL_SIZE = 5
INSTRUCTIVE_POOL_SIZE = 5
NUM_EXAMPLES = 3

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index = pinecone.GRPCIndex(INDEX_NAME)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

generation_config = GenerationConfig(
    do_sample=True,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    max_new_tokens=MAX_NEW_TOKENS
)

model = LanguageModel(MODEL_PATH, generation_config=generation_config,
                      load_in_8bit=LOAD_IN_8BIT, access_token=HF_TOKEN,
                      system_tag=SYSTEM_TAG, user_tag=USER_TAG, ai_tag=AI_TAG)

dataset = load_dataset(DATASET_NAME, 'rc.nocontext')

prompter = EPS(index, embedding_model, SIMILAR_POOL_SIZE, INSTRUCTIVE_POOL_SIZE)

agent = PWS(model)

sanitize = lambda text: text.strip().lower().translate(str.maketrans('', '', string.punctuation))
em = []

for instance in tqdm(dataset['train']):
    question = instance['question']
    list_of_candidates = [sanitize(alias) for alias in instance["answer"]["aliases"]]

    selection = prompter.select_examples(question, NUM_EXAMPLES)
    examples = [entry['metadata'] for entry in selection]
    response = agent.run(question, examples, verbose=True)
    answer = sanitize(response['output'])

    if answer in list_of_candidates:
        em.append(True)

        for entry in selection:
            entry['metadata']['score'] += 1
            prompter.update_score(entry)

        tools = set()
        for calls in response['planner_response']['tool_calls'].values():
            tool = calls.split('[', 1)[0]
            tools.add(tool)
        tools = list(tools)
        new_entry_metadata = {'question': question,
                              'plan': response['planner_response']['text'],
                              'tools': tools,
                              'dataset_name': DATASET_NAME,
        }
        prompter.upsert_entry(new_entry_metadata)
    else:
        em.append(False)

with open('results.json', 'w') as f:
    json.dump(em, f)
