import json
import math
import os
import string
import sys

import datasets
import pinecone
import torch
import transformers

from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from tqdm.auto import tqdm

# Add EPS dir to sys.path to import functions from EPS/src/
# Looks ugly and breaks PEP8 (E402)
# Unfortunately, there seems to be no better alternative
EPS_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(os.path.dirname(EPS_DIR))

from EPS.src.nodes import Extractor, PWS  # noqa: E402
from EPS.src.utils import LMWrapper, EPS  # noqa: E402

with open('../secrets.json', 'r') as f:
    secrets = json.load(f)

# Define Global Variables
PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
INDEX_NAME = 'plans'

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

MODEL_PATH = "stabilityai/StableBeluga-13B"
HF_TOKEN = None
LOAD_IN_8BIT = True
DEVICE_COUNT = 'auto'

SYSTEM_TAG = "### System:\n"
USER_TAG = "### User:\n"
AI_TAG = "### Assistant:\n"

TEMPERATURE = 0.01
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 256

DATASET_NAME = "trivia_qa"

NUM_EXAMPLES = 3
BATCH_SIZE = 100

RESULTS_DIR = 'results/'


# Define helper functions
def sanitize(text):
    return text.strip().lower().translate(
        str.maketrans('', '', string.punctuation)
    )


def get_path(b_id):
    return f"{RESULTS_DIR}results_batch_{b_id}.json"


# Define the main training function
def main():
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.print('Accelerator initialized...')
    process_id = accelerator.process_index
    num_processes = accelerator.num_processes

    # Silence all but the main process
    if not accelerator.is_main_process:
        datasets.logging.disable_progress_bar()
        transformers.logging.disable_progress_bar()

    # Load the language model and initialize related components
    # Get available GPU memory
    cuda_per_proc = int(torch.cuda.device_count() / num_processes)
    max_memory = {}
    for i in range(cuda_per_proc):
        device_id = process_id * cuda_per_proc + i
        device_memory = torch.cuda.get_device_properties(
            device_id
        ).total_memory
        # Leave 10% free for generation overhead
        max_memory[device_id] = round(device_memory * 0.9)
    accelerator.print(
        f"Loading {MODEL_PATH} {'in 8bit' if LOAD_IN_8BIT else ''}..."
    )
    with accelerator.main_process_first():  # Download only once
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map='auto',
            load_in_8bit=LOAD_IN_8BIT, max_memory=max_memory
        )
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS
    )
    lm_wrapper = LMWrapper(
        model, tokenizer, generation_config,
        system_tag=SYSTEM_TAG, user_tag=USER_TAG, ai_tag=AI_TAG
    )
    agent = PWS(lm_wrapper)
    extractor = Extractor(lm_wrapper)

    # Initialize the evolutionary prompter
    accelerator.print("Initializing database connection...")
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pinecone.GRPCIndex(INDEX_NAME)
    accelerator.print("Initializing embedding models...")
    with accelerator.main_process_first():  # Download only once
        device = f"cuda:{process_id * cuda_per_proc}"
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL, device=device
        )
    accelerator.print("Initializing evolutionary prompter...")
    prompter = EPS(index, embedding_model)

    # Data preparation
    accelerator.print('Preparing the dataset...')
    with accelerator.main_process_first():  # Download only once
        dataset = datasets.load_dataset(DATASET_NAME, 'rc.nocontext')
    # Breakdown data to chunks
    accelerator.print('Chunking data...')
    dataset_size = dataset['train'].num_rows
    chunk_size = int(math.ceil(dataset_size / num_processes))
    data_start = process_id * chunk_size
    data_end = data_start + chunk_size
    chunk = dataset['train'][data_start: data_end]
    data = zip(chunk['question'], chunk['answer'])

    # Initilize loop variables
    accelerator.print('Starting training loop...')
    batch_id = process_id
    batch_offset = num_processes
    results = []
    # Initialize the results directory only once
    if accelerator.is_main_process:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, (question, answer) in tqdm(enumerate(data), total=chunk_size,
                                      disable=not accelerator.is_main_process):
        # Process and save results for each batch
        if i and not i % BATCH_SIZE:
            acc = sum([result['em'] for result in results]) / BATCH_SIZE
            print(f"Processed batch number {batch_id} with {acc} accuracy.")
            with open(get_path(batch_id), "w") as f:
                json.dump(results, f)
            batch_id += batch_offset
            results = []

        # Select examples using the prompter
        selection = prompter.select_examples(question, NUM_EXAMPLES)
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
        instructions = [
            {
                'id': entry['id'],
                'similarity': entry['score']
            }
            for entry in selection
        ]
        results.append({'em': em, 'instructions': instructions})
        # In case of an exact match, add the new plans to the index
        # and increment the scores of the selected instructions
        if em:
            # Aggregate the tools used for this instance
            tool_calls = response['planner_response']['tool_calls'].values()
            tools = list(set([tool for (tool, _) in tool_calls]))
            # Metadata for the new plans
            new_entry_metadata = {
                'question': question,
                'plan': response['planner_response']['text'],
                'tools': tools,
                'dataset_name': DATASET_NAME,
            }
            # Add new plans to the index
            prompter.upsert_entry(new_entry_metadata)
            # Increment scores of the selected instructions
            prompter.increment_score([entry['id'] for entry in selection])
    # Process and save results for the last batch
    acc = sum([result['em'] for result in results]) / len(results)
    print(f"Processed batch number {batch_id} with {acc} accuracy.")
    with open(get_path(batch_id), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
