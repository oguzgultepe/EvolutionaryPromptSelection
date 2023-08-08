import json
import math
import os
import string
import pinecone
import torch
import time
from threading import Lock, Thread
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

from utils import LanguageModel, EPS, PWS
from nodes import Extractor

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

TEMPERATURE = 0.01
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY= 1.0
MAX_NEW_TOKENS = 256
DEVICE_COUNT = 'auto'

DATASET_NAME = "trivia_qa"

SIMILAR_POOL_SIZE = 10
INSTRUCTIVE_POOL_SIZE = 10
NUM_EXAMPLES = 3

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index = pinecone.GRPCIndex(INDEX_NAME)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

prompter = EPS(index, embedding_model,
               SIMILAR_POOL_SIZE, INSTRUCTIVE_POOL_SIZE)

dataset = load_dataset(DATASET_NAME, 'rc.nocontext')

sanitize = lambda text: text.strip().lower().translate(
    str.maketrans('', '', string.punctuation))


class processorThread(Thread):
    def __init__(self, device_id, data, prompter, lock, batch_offset):
        Thread.__init__(self)
        self.device_id = device_id
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS
        )
        self.model = LanguageModel(MODEL_PATH, generation_config=generation_config,
                                   device_map=self.device_id, load_in_8bit=LOAD_IN_8BIT,
                                   access_token=HF_TOKEN, system_tag=SYSTEM_TAG,
                                   user_tag=USER_TAG, ai_tag=AI_TAG)
        self.data = data
        self.prompter = prompter
        self.lock = lock
        self.batch_offset = batch_offset

    def run(self):
        print(f"Thread {self.device_id} started.\n")
        # Initialize the agent and the extractor
        agent = PWS(self.model)
        extractor = Extractor(self.model)
        # Initilize loop variables
        batch_id = self.device_id
        results = []
        for i, (question, answer) in enumerate(self.data):
            # Process and save results for each batch
            if i and not i % 100:
                acc = sum([result['em'] for result in results]) / 100
                print(f"Processed batch number {batch_id} with {acc} accuracy.")
                with open(f"results/results_batch_{batch_id}.json", "w") as f:
                    json.dump(results, f)
                batch_id += self.batch_offset
                results = []
            # Select examples using the prompter
            self.lock.acquire()
            selection = self.prompter.select_examples(question, NUM_EXAMPLES)
            self.lock.release()
            # Run the agent
            examples = [entry['metadata'] for entry in selection]
            response = agent.run(question, examples, verbose=True)
            # Check the correctness of the answer
            list_of_candidates = [sanitize(alias) for alias in answer["aliases"]]
            if sanitize(response['output']) in list_of_candidates:
                em = True
            else:
                # Try extracting the answer from the output
                extracted_output = extractor(response['output'], question)
                if sanitize(extracted_output) in list_of_candidates:
                    em = True
                else:
                    em = False
            instructions = [{'id': entry['id'],
                             'similarity': entry['score']
                            }
                            for entry in selection]
            results.append({'em': em, 'instructions': instructions})

            # In case of an exact match, add the new plans to the index
            # and increment the scores of the selected instructions
            if em:
                # Aggregate the tools used for this instance
                tools = set()
                for calls in response['planner_response']['tool_calls'].values():
                    tool = calls.split('[', 1)[0]
                    tools.add(tool)
                tools = list(tools)
                # Metadata for the new plans
                new_entry_metadata = {'question': question,
                                      'plan': response['planner_response']['text'],
                                      'tools': tools,
                                      'dataset_name': DATASET_NAME,
                }
                self.lock.acquire()
                # Add new plans to the index 
                self.prompter.upsert_entry(new_entry_metadata)
                # Increment scores of the selected instructions
                for entry in selection:
                    self.prompter.increment_score(entry['id'])
                self.lock.release()
        # Process and save results for the last batch    
        acc = sum([result['em'] for result in results]) / len(results)
        print(f"Processed batch number {batch_id} with {acc} accuracy.")
        with open(f"results/results_batch_{batch_id}.json", "w") as f:
            json.dump(results, f)


    def join(self):
        Thread.join(self)


if DEVICE_COUNT == 'auto':
    DEVICE_COUNT = torch.cuda.device_count()

dataset_size = dataset['train'].num_rows
chunk_size = int(math.ceil(dataset_size / DEVICE_COUNT))
chunks = [dataset['train'][(device * chunk_size):((device + 1) * chunk_size)]
          for device in range(DEVICE_COUNT)]
data = [zip(chunk['question'], chunk['answer']) for chunk in chunks]

lock = Lock()

threads = []
for device_id in range(DEVICE_COUNT):
    threads.append(processorThread(device_id, data[device_id], prompter, lock, DEVICE_COUNT))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
