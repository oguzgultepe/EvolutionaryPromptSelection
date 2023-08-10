import json
import os
import pathlib
import re
import wikipedia

# Check if we are running on kaggle
if os.environ.get('KAGGLE_URL_BASE',''):
    PROMPTS_PATH = '/kaggle/input/pws-prompts/prompts.json'
else:
    PROMPTS_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                'prompts.json')

with open(PROMPTS_PATH) as f:
    prompts = json.load(f)
    PLANNER_PROMPT = prompts['PLANNER_PROMPT']
    SOLVER_PROMPT = prompts['SOLVER_PROMPT']
    TOOLS_PROMPT = prompts['TOOLS_PROMPT']
    EXTRACTOR_PROMPT = prompts['EXTRACTOR_PROMPT']

class Node:
    """Basic node class"""
    def __init__(self):
        raise NotImplementedError

    def run (self, inputs):
        raise NotImplementedError


class LLMNode(Node):
    """A node that is based on an LLM"""
    def __init__(self, model):
        self.model = model
        self.system_tag = model.system_tag
        self.user_tag = model.user_tag
        self.ai_tag = model.ai_tag
        self.stops = ['.', '\n']

    def call_llm(self, prompt):
        """Calls the underlying LLM with the given inputs
        Parameters:
        ------------
        prompt: str
            prompt for the LLM

        Returns:
        ------------
        response: str
            LLM response
        """
        response = self.model.generate(prompt, self.stops)
        return response


class Planner(LLMNode):
    """Planner node for making plans within the PWS framework"""
    def __init__(self, model):
        super().__init__(model)
        self.stops = ['\n\n']
        self.prefix = PLANNER_PROMPT['prefix']
        self.suffix = PLANNER_PROMPT['suffix']
        self.tools = TOOLS_PROMPT

    def run(self, task, examples):
        """Generate plans for the given task, examples and tools
        Parameters:
        ------------
        task: str
            Task for which the plan is to be generated
        examples: list(dict)
            Examples related to the task for the fewshot prompt

        Returns:
        ------------
        planner_response: dict(str:obj)
            Planner response contains the plans and the evidences
        """
        prompt = self.generate_prompt(task, examples)
        response = self.call_llm(prompt)
        plans, tool_calls = self.parse_response(response)
        planner_response = {'plans': plans, 'tool_calls': tool_calls,
                            'text':response}
        return planner_response

    def generate_prompt(self, task, examples):
        """Generates a planner prompt for the given task, examples and tools
        Parameters:
        ------------
        task: str
            Task for which the plan is to be generated
        examples: list(dict)
            Examples related to the task for the fewshot prompt

        Returns:
        ------------
        prompt: str
            planner prompt
        """
        tools = {tool: self.tools[tool] for example in examples
                 for tool in example['tools']}

        prompt = f"{self.system_tag}{self.prefix}\n"
        prompt += "Tools can be one of the following:\n"
        for tool, description in tools.items():
            prompt += f"{tool}[input]: {description}\n"
        prompt += f"{self.suffix}\n\n"
        for example in examples:
            prompt += f"{self.user_tag}{example['question'].strip()}\n\n"
            prompt += f"{self.ai_tag}{example['plan'].strip()}\n\n"
        prompt += f"{self.user_tag}{task.strip()}\n\n"
        prompt += self.ai_tag
        return prompt

    def parse_response(self, response):
        """Parse the planner response and return plans and evidences dictionary
        Parameters:
        ------------
        response: str
            Planner response

        Returns:
        ------------
        plans: list(str)
            List that contains the plans
        evidences: dict(str:str)
            Evidence dict conatining evidences and associated tool calls
        """
        plans = []
        tool_calls = {}
        for line in response.splitlines():
            if line.startswith("Plan:"):
                plans.append(line)
            elif len(line) < 3:
                continue
            elif line.startswith("#") and line[1] == "E" and line[2].isdigit():
                e, tool_call = line.split("=", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    tool_calls[e] = tool_call
                else:
                    tool_calls[e] = "No evidence found"
        return plans, tool_calls


class WikipediaWorker(Node):
    """Worker that searches Wikipedia"""
    def __init__(self):
        pass

    def run(self, inputs):
        """Searches Wikipedia for the given inputs and returns the first
        2000 characters of the first page in the search results
        Parameters:
        ------------
        inputs: str
            String input for Wikipedia search

        Returns:
        ------------
        evidence: str
            First paragraph of the first page from the search results
        """
        evidence = "No evidence found."
        pages = wikipedia.search(inputs[:300], results=1)
        if pages:
            try:
                evidence = wikipedia.page(pages[0], auto_suggest=False).content
                evidence = evidence[:2000]
            except:
                pass

        return evidence


class LLMWorker(LLMNode):
    """LLM node to be used for worker calls"""
    def run(self, inputs):
        """Run the LLM as a tool call
        Parameters:
        ------------
        inputs: str
            Input for the tool call

        Returns:
        ------------
        evidence: str
            Cleaned response from the tool call
        """
        # Truncate input if necessary
        tokens = self.model.tokenizer(inputs)['input_ids']
        if len(tokens) > 2000:
            inputs = self.model.tokenizer.decode(tokens[:2000],
                                                 skip_special_tokens=True)
        prompt = self.system_tag
        prompt += "Directly answer the following question with no extra words."
        prompt += f"\n\n{self.user_tag}{inputs.strip()}\n\n{self.ai_tag}"
        response = self.call_llm(prompt)
        evidence = response.strip()
        return evidence


class Worker(Node):
    """Worker node that calls appropriate workers for each tool call"""
    def __init__(self, model):
        self.wiki_worker = WikipediaWorker()
        self.llm_worker = LLMWorker(model)

    def run(self, inputs):
        """Faciliates all tool calls and returns evidences
        Parameters:
        ------------
        inputs: dict(str:str)
            A dictionary of evidence variables and associated tool calls

        Returns:
        ------------
        evidences: dict(str:str)
            A dictinary of evidence variables and the outputs of the associated
            tool calls
        """
        evidences = {}
        for e, tool_call in inputs.items():
            # Do not process tools without input
            if "[" not in tool_call:
                evidences[e] = tool_call
                continue

            # Seperate tool and tool input
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1]

            # Find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in evidences:
                    try:
                        evidence = evidences[var]
                    except KeyError:
                        evidence = "No evidence found."
                    tool_input = tool_input.replace(var, f"[{evidence}]")

            match tool:
                case "Wikipedia":
                    evidences[e] = self.wiki_worker.run(tool_input)
                case "LLM":
                    evidences[e] = self.llm_worker.run(tool_input)
                case _:
                    evidences[e] = "No evidence found."

        return evidences


class Solver(LLMNode):
    """Solver node that solves tasks for given plans and evidences"""
    def __init__(self, model):
        super().__init__(model)
        self.prefix = SOLVER_PROMPT['prefix']
        self.suffix = SOLVER_PROMPT['suffix']

    def run(self, task, plans, evidences):
        """Solve the task based on the given plans and evidences
        Parameters:
        ------------
        task: str
            Task to be solved
        plans: list(str)
            List of plans generated by Planner
        evidences: dict(str:str)
            Dictionary of evidences generated by the Worker

        Returns:
        ------------
        output: str
            Solution generated based on the given plans and evidences
        """
        prompt = f"{self.system_tag}{self.prefix}\n\n"
        prompt += f"{self.user_tag}{task.strip()}\n"
        for i in range(len(plans)):
            e = f"#E{i + 1}"
            plan = plans[i]
            try:
                evidence = evidences[e]
            except KeyError:
                evidence = "No evidence found."
            # Only include the first 500 characters of each evidence
            prompt += f"{plan}\nEvidence: {evidence[:500]}...\n"
        prompt += f"{self.suffix + task.strip()}\n\n{self.ai_tag}"
        output = self.call_llm(prompt)
        return output

class Extractor(LLMNode):
    def __init__(self, model):
        super().__init__(model)
        self.prefix = EXTRACTOR_PROMPT['prefix']

    def __call__(self, statement, question):
        prompt = f"{self.system_tag}{self.prefix}\n"
        prompt += f"{self.user_tag}Statement: {statement}\n"
        prompt += f"Question: {question}\n{self.ai_tag}"
        output = self.call_llm(prompt)
        return output
