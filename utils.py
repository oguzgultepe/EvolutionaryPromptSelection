import re
import time
import torch
from numpy.random import choice
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from nodes import Planner, Worker, Solver


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(self, sequence, tokenizer, initial_decoder_input_length):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids = input_ids[0][self.initial_decoder_input_length:][-self.sequence_id_len:]
        lookback_tokens = self.tokenizer.decode(lookback_ids)
        return self.sequence in lookback_tokens


class LanguageModel:
    """Language model wrapper to be used in nodes"""
    def __init__(self, model_path, generation_config, load_in_8bit=False, access_token=None,
                 system_tag='\n', user_tag='\n', ai_tag='\n'):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_auth_token=access_token)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
            load_in_8bit=load_in_8bit, use_auth_token=access_token)
        self.generation_config = generation_config
        self.system_tag = system_tag
        self.user_tag = user_tag
        self.ai_tag = ai_tag

    def stop_sequences_criteria(self, stop_sequences, initial_decoder_input_length):
        return StoppingCriteriaList(
            [
                MultiTokenEOSCriteria(sequence, self.tokenizer, initial_decoder_input_length)
                for sequence in stop_sequences
            ]
        )

    def generate(self, prompt, stops):
        """Generate text based on given prompt
        Parameters:
        ------------
        prompt: str
            Prompt for the LLM

        Returns:
        ------------
        output_text: str
            LLM generated response
        """
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = input_tokens['input_ids'].shape[1]
        stopping_criteria = self.stop_sequences_criteria(stops, input_length)
        with torch.no_grad():
            output_tokens = model.model.generate(
                **input_tokens,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria
                )

        output_text = model.tokenizer.decode(output_tokens[0][input_length:],
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
    # TODO add comments
    def __init__(self, index, embedding_model, similar_pool_size=5, instructive_pool_size=5):
        self.index = index
        index_stats = self.index.describe_index_stats()
        self.index_size = index_stats['total_vector_count']
        self.dimension = index_stats['dimension']
        self.embedding_model = embedding_model
        self.similar_pool_size = similar_pool_size
        self.instructive_pool_size = instructive_pool_size
        self.most_instructive = []
        self.set_most_instructive()

    def set_most_instructive(self):
        batch_size = 1000
        score = lambda entry: entry['metadata']['score']
        for i in range(0, self.index_size, batch_size):
            # find end of batch
            i_end = min(i+batch_size, self.index_size)
            # create IDs batch
            ids = list(range(i, i_end))
            batch = self.index.query(self.dimension * [0],
                                     top_k=batch_size,
                                     filter={'id':{"$in": ids}},
                                     include_metadata=True)['matches']
            batch_sorted = sorted(batch + self.most_instructive, key=score, reverse=True)
            self.most_instructive = batch_sorted[:self.instructive_pool_size]

    def select_examples(self, task, num_examples=3):
        task_embedding = self.embedding_model.encode(task, show_progress_bar=False).tolist()
        most_similar = self.index.query(task_embedding,
                                        top_k=self.similar_pool_size,
                                        include_metadata=True)['matches']
        instructive_ids = [entry['metadata']['id'] for entry in self.most_instructive]
        most_instructive = self.index.query(task_embedding,
                                            top_k=self.instructive_pool_size,
                                            filter={'id':{"$in": instructive_ids}},
                                            include_metadata=True)['matches']
        pool = most_similar + most_instructive
        weights = [(entry['score'] + 1.0) * entry['metadata']['score'] for entry in pool]
        probabilities = list(map(lambda weight: weight/sum(weights), weights))
        sample_ids = choice(range(len(pool)), num_examples, replace=False, p=probabilities)
        examples = [pool[i] for i in sample_ids]
        return examples

    def update_score(self, entry):
        score = lambda entry: entry['metadata']['score']
        self.index.update(id=entry['id'], set_metadata={"score": score(entry)})
        if score(entry) > score(self.most_instructive[-1]):
            self.most_instructive.append(entry)
            self.most_instructive = sorted(self.most_instructive, key=score, reverse=True)
            self.most_instructive[:self.instructive_pool_size]

    def upsert_entry(self, metadata):
        entry_id = self.index_size
        embedding = self.embedding_model.encode(metadata['question'])
        metadata['id'] = entry_id
        metadata['score'] = 1
        index.upsert(zip([str(entry_id)], [embedding], [metadata]))
        self.index_size += 1
