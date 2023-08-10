import math
import re
import time
import torch
from numpy.random import choice
from transformers import LlamaForCausalLM, LlamaTokenizer
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
        if device_map == "auto":
            self.device = "cuda"
        elif isinstance(device_map, int):
            self.device = f"cuda:{device}"
        else:
            self.device = device_map
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
        input_tokens = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)
        input_length = input_tokens['input_ids'].shape[1]
        stopping_criteria = self.stop_sequences_criteria(stops, input_length)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria
                )
        output_text = self.tokenizer.decode(output_tokens[0][input_length:],
                                            skip_special_tokens=True)
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
    def __init__(self, index, embedding_model):
        self.index = index
        self.embedding_model = embedding_model

    def get_size(self):
        """Returns the total vector count in the index
        Returns:
        ------------
            size: int
                total vector count in the index
        """
        size = self.index.describe_index_stats()['total_vector_count']
        return size

    def select_examples(self, task, num_examples=3, top_k=50):
        """Select instructive examples based on a given task
        This method samples instructions from a pool of examples
        The pool is curated querying the index for the top_k most similar tasks
        The examples are then sampled based on their combined example score:
        ((semantic similarity + 1.0) * (log(instruction_score) + 1.0)
        Parameters:
        ------------
        task: str
            Task for which the instructive examples are to be selected
        nun_examples: int, default=3
            Number of instructive examples to return
        top_k: int
            Example pool size
        Returns:
        ------------
        examples: list(str)
            List of instructive examples relevant to the task
        """
        def combined_score(entry):
            similarity = entry['score']
            score = entry['metadata']['score']
            combined_score = (similarity + 1.0) * (math.log(score) + 1.0)
            return combined_score

        task_embedding = self.embedding_model.encode(
            task, show_progress_bar=False).tolist()
        pool = self.index.query(vector=task_embedding, top_k=top_k,
                                include_metadata=True)['matches']
        weights = [combined_score(entry) for entry in pool]
        probabilities = list(map(lambda weight: weight/sum(weights), weights))
        sample_ids = choice(range(len(pool)), num_examples,
                            replace=False, p=probabilities)
        examples = [pool[i] for i in sample_ids]
        return examples

    def increment_score(self, entry_ids):
        """Increment the instruction score of an example
        Parameters:
        ------------
        entry_ids: str or list(str)
            Increment the score/s of the example/s with given entry_id/s
        """
        score = lambda entry: entry['metadata']['score']
        # Wrap str input with a list
        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]
        # Fetch the entries from the index 
        entries = self.index.fetch(entry_ids)['vectors']
        for entry_id, entry in entries.items():
            self.index.update(id=entry_id,
                              set_metadata={"score": score(entry) + 1})

    def upsert_entry(self, metadata):
        """Upsert a new entry into the index
        Parameters:
        ------------
        metadata: dict
            a dictionary containing entry metadata
            {question, plan, tools, dataset_name}
        """
        embedding = self.embedding_model.encode(
            metadata['question'], show_progress_bar=False).tolist()
        metadata['score'] = 1
        entry_id = self.get_size()
        metadata['id'] = entry_id
        self.index.upsert(zip([str(entry_id)], [embedding], [metadata]))
