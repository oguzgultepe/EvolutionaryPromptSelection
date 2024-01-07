import math
import torch
import numpy as np
import pandas as pd

from numpy.random import choice
from scipy import stats
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList


class MultiTokenEOSCriteria(StoppingCriteria):
    """
    Stopping criteria based on a given multi-token sequence.
    Please refer to HuggingFace Transformers library for documentation
    """
    def __init__(self, sequence, tokenizer, initial_decoder_input_length):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        For efficiency, we compare the last n tokens where n is the
        number of tokens in the stop_sequence
        """
        lookback_ids = input_ids[0][self.initial_decoder_input_length:]
        lookback_ids = lookback_ids[-self.sequence_id_len:]
        lookback_tokens = self.tokenizer.decode(lookback_ids)
        return self.sequence in lookback_tokens


class LMWrapper:
    """Language model wrapper to be used in nodes"""
    def __init__(self, model, tokenizer, generation_config,
                 system_tag='\n', user_tag='\n', ai_tag='\n'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = f"cuda:{min(model.hf_device_map.values())}"
        self.generation_config = generation_config
        self.system_tag = system_tag
        self.user_tag = user_tag
        self.ai_tag = ai_tag

    def stop_sequences_criteria(self, stop_sequences,
                                initial_decoder_input_length):
        """
        Creates a custom stopping criteria for the given input
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
        """
        Generate text based on given prompt
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
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        input_tokens = input_tokens.to(self.device)
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


class EPS:
    """ Evolutionary Prompt Selection"""
    def __init__(self, index, embedding_model):
        self.index = index
        self.embedding_model = embedding_model

    def get_size(self):
        """
        Returns the total vector count in the index
        Returns:
        ------------
            size: int
                total vector count in the index
        """
        size = self.index.describe_index_stats()['total_vector_count']
        return size

    def select_examples(self, task, num_examples=3, top_k=50):
        """
        Select instructive examples based on a given task
        This method samples instructions from a pool of examples
        The pool is curated querying the index for the top_k most
        similar tasks
        The examples are then sampled based on their combined example
        score:
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
        examples: list(dict)
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
        """
        Increment the instruction score of an example
        Parameters:
        ------------
        entry_ids: str or list(str)
            Increment the score/s of the example/s with given entry_id/s
        """
        # Wrap str input with a list
        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]
        # Fetch the entries from the index
        entries = self.index.fetch(entry_ids)['vectors']
        for entry_id, entry in entries.items():
            self.index.update(
                id=entry_id,
                set_metadata={
                    "score": entry['metadata']['score'] + 1
                }
            )

    def upsert_entry(self, metadata):
        """
        Upsert a new entry into the index
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

    def create_nis(self, show_progress_bar=True):
        """
        Creates a new normalized instruction sampler (NIS) based on the
        instructions in the index
        """
        total_vector_count = self.get_size()
        chunk_size = 1000
        instructions = {}
        embeddings = {}
        for chunk_start in tqdm(range(0, total_vector_count, chunk_size),
                                disable=not show_progress_bar):
            chunk_end = min(chunk_start + 1000, total_vector_count)
            idx = [str(i) for i in range(chunk_start, chunk_end)]
            vectors = self.index.fetch(idx)['vectors']
            for entry_id, entry in vectors.items():
                instructions[int(entry_id)] = entry['metadata']
                embeddings[int(entry_id)] = entry['values']

        instructions = pd.DataFrame.from_dict(instructions, orient='index')
        instructions = instructions.sort_index()
        embeddings = np.array([v for _, v in sorted(embeddings.items())])
        return NIS(instructions, embeddings, self.embedding_model)


class NIS:
    '''Normalized Instruction Sampler'''
    def __init__(
        self, instructions, embeddings, embedding_model, pool_size=50
    ):
        self.instructions = instructions
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.pool_size = pool_size
        log_index = np.log(instructions.index + 1)
        lr = stats.linregress(log_index, instructions.score)
        log_curve = (lr.slope * log_index + lr.intercept)
        self.normalized_scores = instructions.score / log_curve

    def sample_instructions(self, task, num_instructions=3,
                            sampling='pool-prob', adjust_scores=True):
        '''
        Sample instructions based on a task
        Parameters
        ------------
        task: str
            Task in natural language
        num_instructions: int, default=3
            Number of instructions to be samples
        sampling: str, default='pool-prob'
            Sampling strategy
            'random': randomly sample from all instructions
            'pool-prob': curate a pool of similar instructions
                         then sample instructions based on score
            'pool-top': curate a pool of similar instructions
                        then select instructions with score
            'pool-random': curate a pool of similar instructions
                           then sample randomly
        adjust_scores: bool, default=True
            If True, scores are multiplied by similarity values

        Returns
        ------------
        : pandas Series
            contains sampled instructions
        '''
        # Directly sample if sampling == 'random'
        if sampling == 'random':
            selected_instructions = self.instructions.sample(
                n=num_instructions)
        else:
            # Encode question
            question_embedding = self.embedding_model.encode(
                task, normalize_embeddings=True, show_progress_bar=False)
            # Compute similarity
            similarities = self.embeddings @ question_embedding
            # Select similar pool
            pool_ids = similarities.argpartition(kth=self.pool_size)
            pool_ids = pool_ids[:self.pool_size]
            pool_scores = self.normalized_scores[pool_ids]
            # Multiply scores by (semantic similarity + 1)
            if adjust_scores:
                pool_similarities = similarities[pool_ids]
                pool_scores *= (pool_similarities + 1)
            match sampling:
                # Similarity pool + probabilistic sampling
                case 'pool-prob':
                    instruction_ids = pool_scores.sample(n=num_instructions,
                                                         weights=pool_scores)
                # Similarity pool + select top scorers
                case 'pool-top':
                    instruction_ids = pool_scores.sort_values(ascending=False)
                    instruction_ids = instruction_ids[:num_instructions]
                # Similarity pool + random sampling
                case 'pool-random':
                    instruction_ids = pool_scores.sample(n=num_instructions)
            # Select instructions
            instruction_ids = instruction_ids.index
            selected_instructions = self.instructions.loc[instruction_ids]
            # Make sure it is a copy and not a view
            selected_instructions = selected_instructions.copy()
            # Include normalized/adjusted scores
            selected_instructions.score = pool_scores[instruction_ids]
        # Convert to list of dictionaries format
        selected_instructions = selected_instructions.to_dict(orient='index')
        return list(selected_instructions.values())
