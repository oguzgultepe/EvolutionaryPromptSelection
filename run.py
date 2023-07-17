from transformers import GenerationConfig

from utils import LanguageModel, EPS, PWS

MODEL_PATH = "lmsys/vicuna-7b-v1.3"
TEMPERATURE = 0.7
TOP_K = 100
TOP_P = 0.7
REPETITION_PENALTY= 1.0
MAX_NEW_TOKENS = 512

generation_config = GenerationConfig(
    do_sample=True,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    max_new_tokens=MAX_NEW_TOKENS
)

model = LanguageModel(MODEL_PATH, generation_config=generation_config)
prompter = EPS()
agent = PWS(model)

while(True):
    task = input("Enter your question...")
    examples, tools = prompter.select_examples(task)
    print(agent.run(task, examples, tools))


