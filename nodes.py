import json
import os
import pathlib
import wikipedia

PROMPTS_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'prompts')
with open(os.path.join(PROMPTS_DIR, 'planner.json'), 'r') as f:
    PLANNER_PROMPT = json.load(f)
with open(os.path.join(PROMPTS_DIR, 'solver.json'), 'r') as f:
    SOLVER_PROMPT = json.load(f)

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

    def run(self, inputs):
        """Calls the underlying LLM with the given inputs
        Parameters:
        ------------
        inputs: str
            inputs for the LLM

        Returns:
        ------------
        response: str
            LLM response
        """
        response = self.model.generate(inputs)
        return response

class Planner(LLMNode):
    """Planner node for making plans within the PWS framework"""
    def __init__(self, model):
        self.prefix = PLANNER_PROMPT['prefix']
        self.suffix = PLANNER_PROMPT['suffix']
        self.model = model

    def run(self, task, examples, tools):
        """Generate plans for the given task, examples and tools
        Parameters:
        ------------
        task: str
            Task for which the plan is to be generated
        examples: list(str)
            Examples related to the task for the fewshot prompt
        tools: dict(str:str)
            Tools that can be used to solve the task

        Returns:
        ------------
        planner_response: dict(str:obj)
            Planner response contains the plans and the evidences
        """
        prompt = generate_prompt(task, examples, tools)
        response = super.run(prompt)
        plans, tool_calls = parse_response(response)
        planner_response = {'plans': plans, 'tool_calls': tool_calls}
        return planner_response

    def generate_prompt(task, examples, tools):
        """Generates a planner prompt for the given task, examples and tools
        Parameters:
        ------------
        task: str
            Task for which the plan is to be generated
        examples: list(str)
            Examples related to the task for the fewshot prompt
        tools: dict(str:str)
            Tools that can be used to solve the task

        Returns:
        ------------
        prompt: str
            planner prompt
        """
        prompt = self.prefix
        prompt += generate_worker_prompt(tools)
        prompt += examples.join('\n')
        prompt += self.suffix
        prompt += task + '\n'
        return prompt

    def generate_worker_prompt(tools):
        """Generates a worker prompt for given tools
        Parameters:
        ------------
        tools: dict(str:str)
            contains the names and descriptions of the tools

        Returns:
        ------------
        prompt: str
            worker prompt
        """
        prompt = "Tools can be one of the following:\n"
        for tool, description in tools.items():
            prompt += f"{tool}[input]: {description}\n"
        return prompt + "\n"

    def parse_response(response):
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
    def run(self, inputs):
        """Searches Wikipedia for the given inputs and returns the first
        paragraph of the first page in search results
        Parameters:
        ------------
        inputs: str
            String input for Wikipedia search

        Returns:
        ------------
        evidence: str
            First paragraph of the first page from the search results
        """
        page = wikipedia.search(inputs, results=1)[0]
        content = wikipedia.page(page, auto_suggest=False).content
        evidence = content.split('\n\n', 1)[0]
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
        prompt = f"Respond in short directly with no extra words.\n\n{inputs}"
        response = super.run(prompt)
        evidence = response.strip("\n")
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
                    tool_input = tool_input.replace(var, f"[{evidences[var]}]")

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
            self.prefix = SOLVER_PROMPT['prefix']
            self.suffix = SOLVER_PROMPT['suffix']
            self.model = model

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
        prompt = self.prefix
        prompt += task + '\n'
        for i in range(len(plans)):
            e = f"#E{i + 1}"
            prompt += f"{plans[i]}\nEvidence:\n{evidences[e]}\n"
        prompt += self.suffix
        prompt += task + '\n'
        output = super.run(prompt)
        return output
