from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import ragas
import yaml
import os

# Load configuration
_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
model_name_default = _config.get('model_name', 'llama3.2:3b')

ragas.llm = OllamaLLM(model=model_name_default)
ragas.embeddings = OllamaEmbeddings(model=model_name_default)

data = {
    "question": [
        "What is the punishment for theft?",
        "Can a person be arrested without a warrant?",
        "What is the right to information?"
    ],
    "answer": [
        "The punishment for theft under Section 379 of IPC 1860 is imprisonment up to three years, or a fine, or both. Theft is also addressed in Section 303 of the BNS 2023. [Citation 1]",
        "Yes, under Section 35 of the BNSS 2023, a police officer can arrest a person without a warrant if they are involved in a cognizable offence. [Citation 1]",
        "The Right to Information Act 2005 gives citizens the right to request access to government information by filing an application. [Citation 1]"
    ],
    "contexts": [
        ["Section 378 of the Indian Penal Code (IPC) 1860 defines theft. Section 379 prescribes punishment for theft which is imprisonment of either description for a term which may extend to three years, or with fine, or with both. The Bharatiya Nyaya Sanhita (BNS) 2023 also covers theft under Section 303."],
        ["Under the Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, specifically Section 35, any police officer may without an order from a Magistrate and without a warrant, arrest any person who has been concerned in any cognizable offence."],
        ["The Right to Information Act, 2005 mandates timely response to citizen requests for government information. It is an initiative taken by Department of Personnel and Training, Ministry of Personnel, Public Grievances and Pensions."]
    ],
    "ground_truth": [
        "The punishment for theft is up to three years imprisonment or fine or both under IPC. It is also regulated under BNS Section 303.",
        "Yes, under BNSS 2023 Section 35, a police officer can arrest a person without a warrant for a cognizable offence.",
        "Information rights mandate timely response to citizen requests for information."
    ]
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)
