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

MODEL_PATH = "stabilityai/StableBeluga-13B"
SYSTEM_TAG = "### System:\n"
USER_TAG = "### User:\n"
AI_TAG = "### Assistant:\n"

LOAD_IN_8BIT = True
HF_TOKEN = secrets["HF_TOKEN"]

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
extractor = Extractor(model)
em = []
prompt_data = []
instance_counter = 0
for instance in tqdm(dataset['train']):
    if instance_counter and not instance_counter % 100:
        total_acc = sum(em) / len(em)
        last_100_acc = sum(em[-100:]) / 100
        print(f"Processed instances: {instance_counter}")
        print(f"\t{total_acc=}\t{last_100_acc=}")
        results = {'em': em[-100:], 'prompt_data': prompt_data[-100:]}
        batch_number = instance_counter / 100
        with open(f"results_batch_{batch_number}.json", "w") as f:
            json.dump(results, f)

    instance_counter += 1
    question = instance['question']
    list_of_candidates = [sanitize(alias) for alias in instance["answer"]["aliases"]]
    selection = prompter.select_examples(question, NUM_EXAMPLES)
    prompt_data.append([(entry['id'], entry['score']) for entry in selection])
    examples = [entry['metadata'] for entry in selection]
    response = agent.run(question, examples, verbose=True)
    answer = sanitize(response['output'])

    if answer not in list_of_candidates:
        extracted = sanitize(extractor(response['output'], question))
        if extracted not in list_of_candidates:
            em.append(False)
            continue
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
