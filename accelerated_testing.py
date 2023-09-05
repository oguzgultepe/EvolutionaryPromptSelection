import argparse
import json
import math
import os
import re
import string

import datasets
import pinecone
import torch
import transformers

from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from tqdm.auto import tqdm

from nodes import DirectPrompter, PWS
from utils import LMWrapper, EPS

with open('secrets.json', 'r') as f:
    secrets = json.load(f)

# Define Global Variables
PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
INDEX_NAME = 'plans'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VALID_DATASETS = {'trivia_qa', 'hotpot_qa'}
VALID_SAMPLINGS = {'direct', 'random', 'pool-random', 'pool-prob', 'pool-top'}
RESULTS_DIR = 'test_results/'


# Define evaluation functions
def trivia_em(prediction, answer):
    def sanitize(text):
        return text.strip().lower().translate(
            str.maketrans('', '', string.punctuation)
        )

    candidates = answer["aliases"]
    sanitized_prediction = sanitize(prediction)
    sanitized_candidates = [sanitize(answer) for answer in candidates]
    return sanitized_prediction in sanitized_candidates


def hotpot_em(prediction, answer):
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    return (normalize_answer(prediction) == normalize_answer(answer))


def run_tests(TEST_CONFIG_PATH):
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.print("Accelerator initialized...")
    process_id = accelerator.process_index
    num_processes = accelerator.num_processes

    # Silence all but the main process
    if not accelerator.is_main_process:
        datasets.logging.disable_progress_bar()
        transformers.logging.disable_progress_bar()

    # Read and process test config
    accelerator.print(f"Loading test config: \'{TEST_CONFIG_PATH}\'...")
    try:
        with open(TEST_CONFIG_PATH, 'r') as f:
            TEST_CONFIG = json.load(f)
        MODEL_CONFIG = TEST_CONFIG['MODEL_CONFIG']
        MODEL_PATH = MODEL_CONFIG['MODEL_PATH']
        LOAD_IN_8BIT = MODEL_CONFIG['LOAD_IN_8BIT']
        SYSTEM_TAG = MODEL_CONFIG['SYSTEM_TAG']
        USER_TAG = MODEL_CONFIG['USER_TAG']
        AI_TAG = MODEL_CONFIG['AI_TAG']

        GENERATION_CONFIG = TEST_CONFIG['GENERATION_CONFIG']
        TEMPERATURE = GENERATION_CONFIG['TEMPERATURE']
        TOP_K = GENERATION_CONFIG['TOP_K']
        TOP_P = GENERATION_CONFIG['TOP_P']
        REPETITION_PENALTY = GENERATION_CONFIG['REPETITION_PENALTY']
        MAX_NEW_TOKENS = GENERATION_CONFIG['MAX_NEW_TOKENS']

        TEST_SIZE = TEST_CONFIG['TEST_SIZE']
        if not isinstance(TEST_SIZE, int):
            accelerator.print(f"Invalid test size: {TEST_SIZE}")
            return

        # Ensure the tests are valid
        TESTS = TEST_CONFIG['TESTS']
        for test in TESTS:
            if test['DATASET_NAME'] not in VALID_DATASETS:
                accelerator.print(
                    f"Invalid dataset name: {test['DATASET_NAME']}"
                )
                return
            if test['SAMPLING'] not in VALID_SAMPLINGS:
                accelerator.print(
                    f"Invalid sampling: {test['SAMPLING']}"
                )
                return
            if (
                not isinstance(test['PROMPT_SIZE'], int)
                or test['PROMPT_SIZE'] < 3
                or test['PROMPT_SIZE'] > 12
            ):
                accelerator.print(
                    f"Invalid prompt size: {test['PROMPT_SIZE']}"
                )
                return
            if not isinstance(test['ADJUST_SCORES'], bool):
                accelerator.print(
                    f"Invalid adjust scores: {test['ADJUST_SOCRES']}"
                )
                return

    except FileNotFoundError as e:
        accelerator.print(f"Test config not found: {e}")
        return

    except KeyError as e:
        accelerator.print(f"Incorrect config format: {e}")
        return

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
    direct = DirectPrompter(lm_wrapper)

    # Initialize sampler
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
    accelerator.print("Initializing normalized instruction samplers...")
    sampler = EPS(index, embedding_model).create_nis(
        show_progress_bar=accelerator.is_main_process
    )

    # Data preparation
    accelerator.print(
        f"Preparing test data ({TEST_SIZE} samples)..."
    )
    # Distribute tests to processes
    chunk_size = int(math.ceil(TEST_SIZE / num_processes))
    chunk_start = process_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, TEST_SIZE)
    test_data = {
        test['DATASET_NAME']: None
        for test in TESTS
    }
    for dataset_name in test_data.keys():
        match dataset_name:
            case 'trivia_qa':
                subset = 'rc.nocontext'
            case 'hotpot_qa':
                subset = 'fullwiki'
        accelerator.print(f"Loading {dataset_name}...")
        with accelerator.main_process_first():  # Download only once
            dataset = datasets.load_dataset(dataset_name, subset)
        chunk = dataset['validation'][chunk_start:chunk_end]
        test_data[dataset_name] = (chunk['question'], chunk['answer'])

    # Initialize temporary results directory
    tmp_dir = os.path.join(RESULTS_DIR, 'tmp')
    if accelerator.is_main_process:
        os.makedirs(tmp_dir, exist_ok=True)
        all_results = []

    accelerator.print("Starting tests...\n")
    for test_id, test in enumerate(TESTS):
        # Get test variables
        dataset_name = test['DATASET_NAME']
        sampling = test['SAMPLING']
        adjust_scores = test['ADJUST_SCORES']
        prompt_size = test['PROMPT_SIZE']
        accelerator.print(f"Test {test_id + 1}/{len(TESTS)}:")
        accelerator.print(''.join(['-'] * 64))
        accelerator.print(
            '\t'.join([f"{var}={val}" for var, val in test.items()])
        )

        # Define instruction and prediction function
        if sampling == 'direct':  # Direct prompting
            def get_instructions(question):
                # No instructions for direct prompting
                return None

            def predict(question, instructions):
                return direct(question)

        else:  # Fewshot
            def get_instructions(question):
                instructions = sampler.sample_instructions(
                    question, prompt_size,
                    sampling=sampling,
                    adjust_scores=adjust_scores
                )
                return instructions

            def predict(question, instructions):
                response = agent.run(
                    question, instructions, use_extractor=True
                )
                return response['extracted_output']

        # Get data and evaluation method
        data = test_data[dataset_name]
        match dataset_name:
            case 'trivia_qa':
                check_em = trivia_em
            case 'hotpot_qa':
                check_em = hotpot_em

        # Test begins
        chunk_results = []
        for i, (question, answer) in tqdm(
            enumerate(zip(*data)), total=chunk_size,
            disable=not accelerator.is_main_process
        ):
            instructions = get_instructions(question)
            prediction = predict(question, instructions)
            em = check_em(prediction, answer)
            if instructions:
                instructions = [
                    {
                        'id': int(instruction['id']),
                        'score': instruction['score']}
                    for instruction in instructions
                ]
            chunk_results.append(
                {
                    'q_id': i + process_id * chunk_size,
                    'em': em,
                    'instructions': instructions
                }
            )

        # Save chunk results from each process
        tmp_results_path = os.path.join(
            tmp_dir, f'tmp{process_id}.json'
        )
        with open(tmp_results_path, 'w') as f:
            json.dump(chunk_results, f)

        # Combine results from all processes
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            combined_results = []
            for chunk_id in range(num_processes):
                tmp_results_path = os.path.join(tmp_dir, f'tmp{chunk_id}.json')
                with open(tmp_results_path, 'r') as f:
                    chunk_results = json.load(f)
                combined_results.extend(chunk_results)
                os.remove(tmp_results_path)
            mean_em = sum([r['em'] for r in combined_results]) / TEST_SIZE
            accelerator.print(
                f"Test {test_id + 1}: Mean Exact Match = {mean_em}\n"
            )
            test_results = {
                'CONFIG': test,
                'EM': mean_em,
                'RESULTS': combined_results
            }
            all_results.append(test_results)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Remove the temporary results directory
        os.rmdir(tmp_dir)
        # Save all results
        results_path = os.path.join(
            RESULTS_DIR,
            os.path.basename(TEST_CONFIG_PATH)
        )
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'MODEL_CONFIG': TEST_CONFIG['MODEL_CONFIG'],
                    'TESTS': all_results
                }, f
            )


def main():
    parser = argparse.ArgumentParser(description="Accelerated Testing")
    parser.add_argument("--test-config")
    args = parser.parse_args()
    test_config = args.test_config
    run_tests(test_config)


if __name__ == '__main__':
    main()
