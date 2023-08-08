import math
import re
import time
import torch
from numpy.random import choice
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from nodes import Planner, Worker, Solver


class MultiTokenEOSCriteria(StoppingCriteria):
    """Stopping criteria based on a given multi-token sequence.
    Please refer to HuggingFace Transformers library for documentation"""

    def __init__(self, sequence, tokenizer, initial_decoder_input_length):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number
        # of tokens in the stop_sequence
        lookback_ids = input_ids[0][self.initial_decoder_input_length:]
        lookback_ids = lookback_ids[-self.sequence_id_len:]
        lookback_tokens = self.tokenizer.decode(lookback_ids)
        return self.sequence in lookback_tokens


class LanguageModel:
    """Language model wrapper to be used in nodes"""
    def __init__(self, model_path, generation_config,
                 device_map='auto', load_in_8bit=False, access_token=None,
                 system_tag='\n', user_tag='\n', ai_tag='\n'):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path,
            use_auth_token=access_token)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device_map,
            load_in_8bit=load_in_8bit, use_auth_token=access_token)
        self.generation_config = generation_config
        if device_map == 'auto':
            self.device = 'cuda'
        else:
            self.device = f'cuda:{device_map}'
        self.system_tag = system_tag
        self.user_tag = user_tag
        self.ai_tag = ai_tag

    def stop_sequences_criteria(self, stop_sequences,
                                initial_decoder_input_length):
        """Creates a custom stopping criteria for the given input
        Parameters:
        ------------
        stop_sequences: list(str)
            A list of strings that ends text generation
        initial_decoder_input_length: int
            Total number of tokens in the initial input

        Returns:
        ------------
            StoppingCriteriaList object
        """
        return StoppingCriteriaList(
            [
                MultiTokenEOSCriteria(sequence, self.tokenizer,
                                      initial_decoder_input_length)
                for sequence in stop_sequences
            ]
        )

    def generate(self, prompt, stops):
        """Generate text based on given prompt
        Parameters:
        ------------
        prompt: str
            Prompt for the LLM
        stops: list(str)
            List of strings to be used as stopping criteria

        Returns:
        ------------
        output_text: str
            LLM generated response
        """
        start = time.perf_counter()
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        encoding_time = time.perf_counter() - start
        input_length = input_tokens['input_ids'].shape[1]
        stopping_criteria = self.stop_sequences_criteria(stops, input_length)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria
                )
        gpu_time = time.perf_counter() - (start + encoding_time)
        output_text = self.tokenizer.decode(output_tokens[0][input_length:],
                                            skip_special_tokens=True)
        decoding_time = time.perf_counter() - (start + encoding_time + gpu_time)
        log_message = f"Device {self.device} generated:\n\t{len(output_tokens[0])-input_length} new tokens\n"
        log_message += f"\tfrom: {input_length} prompt tokens\n"
        log_message += f"\tin: {time.perf_counter() - start} seconds\n"
        print(log_message)
        return output_text

class PWS:
    """ Planner Worker Solver Framework"""
    def __init__(self, model):
        self.planner = Planner(model=model)
        self.worker = Worker(model=model)
        self.solver = Solver(model=model)

    def run(self, task, examples, verbose=False):
        """Run the PWS on a given task based on provided examples
        Parameters:
        ------------
        task: str
            Task for which the PWS is to be run
        examples: list(str)
            Examples related to the task for the fewshot prompt
        verbose: bool, default=False
            If True, responses from intermediate nodes are also returned

        Returns:
        ------------
        pws_response: dict(str:obj)
            PWS response contains the output and time elapsed
            If verbose responses from intermediate nodes are also returned
        """

        st = time.time()
        # Plan
        planner_response = self.planner.run(task, examples)
        plans = planner_response["plans"]
        tool_calls = planner_response["tool_calls"]

        # Work
        evidences = self.worker.run(tool_calls)

        # Solve
        output = self.solver.run(task, plans, evidences)

        wall_time = time.time() - st

        pws_response = {"output": output,
                        "wall_time": wall_time}

        if verbose:
            pws_response["planner_response"] = planner_response
            pws_response["worker_response"] = evidences

        return pws_response


class EPS:
    """ Evolutionary Prompt Selection"""
    def __init__(self, index, embedding_model,
                 similar_pool_size=5, instructive_pool_size=5):
        self.index = index
        index_stats = self.index.describe_index_stats()
        self.index_size = index_stats['total_vector_count']
        self.embedding_model = embedding_model
        self.similar_pool_size = similar_pool_size
        self.instructive_pool_size = instructive_pool_size
        self.most_instructive = []
        self.set_most_instructive()

    def set_most_instructive(self):
        """Retrieve the most instructive examples from the index
        Pinecone does not support aggregations over metadata so we fetch all
        instructions and manually select the most instructive ones
        """
        batch_size = 1000
        score = lambda entry: entry['metadata']['score']
        for i in range(0, self.index_size, batch_size):
            # Find end of batch
            i_end = min(i+batch_size, self.index_size)
            # Create IDs batch
            ids = [str(idx) for idx in range(i, i_end)]
            batch = list(self.index.fetch(ids)['vectors'].values())
            # Sort and keep the most instructive
            batch_sorted = sorted(batch + self.most_instructive,
                                  key=score, reverse=True)
            self.most_instructive = batch_sorted[:self.instructive_pool_size]

    def select_examples(self, task, num_examples=3):
        """Select instructive examples based on a given task
        This method samples instructions from a curated pool of examples
        The pool is curated by combinining (similar_pool_size) number of
        semantically similar examples and (instructive_pool_size) number of
        examples with high instruction score
        The examples are then sampled based on their combined example score:
        ((semantic similarity + 1.0) * (log(instruction_score) + 1.0)
        Parameters:
        ------------
        task: str
            Task for which the instructive examples are to be selected
        nun_examples: int, default=3
            Number of instructive examples to return

        Returns:
        ------------
        examples: list(str)
            List of instructive examples relevant to the task
        """
        task_embedding = self.embedding_model.encode(task,
            show_progress_bar=False).tolist()
        most_similar = self.index.query(task_embedding,
                                        top_k=self.similar_pool_size,
                                        include_metadata=True)['matches']
        instructive_ids = [entry['metadata']['id']
                           for entry in self.most_instructive]
        most_instructive = self.index.query(task_embedding,
            top_k=self.instructive_pool_size,
            filter={'id':{"$in": instructive_ids}},
            include_metadata=True)['matches']
        pool = most_similar + most_instructive
        weights = [(entry['score'] + 1.0) * (math.log(
            entry['metadata']['score']) + 1.0) for entry in pool]
        probabilities = list(map(lambda weight: weight/sum(weights), weights))
        sample_ids = choice(range(len(pool)), num_examples,
                            replace=False, p=probabilities)
        examples = [pool[i] for i in sample_ids]
        return examples

    def increment_score(self, entry_id):
        """Increment the instruction score of an example
        Parameters:
        ------------
        entry_id: str
            Score of the example with the given entry_id is incremented
        """
        score = lambda entry: entry['metadata']['score']
        entry = None
        # Check if the entry is in the most instructive pool
        for candidate in self.most_instructive:
            if candidate['id'] == entry_id:
                entry = candidate
        # If the entry is not in the most instructive pool
        if not entry:
            # Fetch it from the index 
            entry = self.index.fetch([entry_id])['vectors'][entry_id]
            # Add the entry to the most instructive pool
            self.most_instructive.append(entry)
        # Update entry score   
        entry['metadata']['score'] += 1
        self.index.update(id=entry['id'], set_metadata={"score": score(entry)})
        # Sort the most instructive pool and keep the most instructive
        self.most_instructive = sorted(self.most_instructive,
                                       key=score, reverse=True)
        self.most_instructive = self.most_instructive[
            :self.instructive_pool_size]

    def upsert_entry(self, metadata):
        """Upsert a new entry into the index
        Parameters:
        ------------
        metadata: dict
            a dictionary containing entry metadata
            {question, plan, tools, dataset_name}
        """
        entry_id = self.index_size
        embedding = self.embedding_model.encode(
            metadata['question'], show_progress_bar=False).tolist()
        metadata['id'] = entry_id
        metadata['score'] = 1
        self.index.upsert(zip([str(entry_id)], [embedding], [metadata]))
        self.index_size += 1
