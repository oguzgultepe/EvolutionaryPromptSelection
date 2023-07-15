import re
import time
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from nodes import Planner, Worker, Solver


class LanguageModel:
    """Language model wrapper to be used in nodes"""
    def __init__(self, model_path, generation_config):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto')
        self.generation_config = generation_config

    def generate(self, prompt):
        """Generate text based on given prompt
        Parameters:
        ------------
        prompt: str
            Prompt for thee LLM

        Returns:
        ------------
        llm_response: str
            LLM generated response
        """
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.LongTensor(tokens).unsqueeze(0)
        tokens = tokens.to('cuda')

        length = len(tokens[0])
        with torch.no_grad():
            rest = self.model.generate(
                input_ids=tokens,
                generation_config = self.generation_config
            )
        output = rest[0][length:]
        llm_response = self.tokenizer.decode(output, skip_special_tokens=True)
        return llm_response

class PWS:
    """ Planner Worker Solver Framework"""
    def __init__(self, model):
        self.planner = Planner(model=model)
        self.worker = Worker(model=model)
        self.solver = Solver(model=model)

    def run(self, task, examples, tools, verbose=False):
        """Run the PWS on a given task based on provided examples/tools
        Parameters:
        ------------
        task: str
            Task for which the PWS is to be run
        examples: list(str)
            Examples related to the task for the fewshot prompt
        tools: dict(str:str)
            Tools that can be used to solve the task

        Returns:
        ------------
        pws_response: dict(str:obj)
            PWS response contains the output and time elapsed
            If verbose responses from intermediate nodes are also returned
        """

        st = time.time()
        # Plan
        planner_response = self.planner.run(task, examples, tools)
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
            pws_response["worker_response"] = worker_evidences

        return pws_response


class EPS:
    """ Evolutionary Prompt Selection"""
    #TODO
    def __init__(self):
        raise NotImplementedError
    def select_examples(self, task, num_examples):
        raise NotImplementedError
        return examples, tools

