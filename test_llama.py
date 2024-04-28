import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from build_dataset import prompt_formatter_qa, prompt_formatter_summ
import numpy as np
import json 
from tqdm import tqdm


model_folder = "llama2-7b-chat-ft/checkpoint-40/"

def load_dataset():
    with open('data/test_qa.json', 'r') as f:
        test_qa = json.load(f)

    with open('data/test_sum.json', 'r') as f:
        test_sum = json.load(f)
    
    return test_qa, test_sum


model = AutoPeftModelForCausalLM.from_pretrained(
    model_folder,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_folder)


def inference(prompt, task):
    if task == 'qa':
        real = prompt['instruction'].split('### Answer:\n')[1].replace('</s>','').strip()
        text = prompt['instruction'].split('### Answer:\n')[0] + '### Answer:\n'
    elif task == 'sum':
        real = prompt['instruction'].split('### Summary:\n')[1].replace('</s>','').strip()
        text = prompt['instruction'].split('### Summary:\n')[0] + '### Summary:\n'

    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, temperature=0.7)

    if task == 'qa':
        return {'real': real,
                'pred': 
                tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].split('### Answer:\n')[1]
        }
    elif task == 'sum':
        return {'real': real,
                'pred':   
                tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].split('### Summary:\n')[1]
        }


def calculate_perplexity(tokenizer, model, dataset):
    total_loss = 0
    total_tokens = 0

    for sample in tqdm(dataset):
        token_ids = tokenizer.encode(sample['instruction'], return_tensors="pt")
        with torch.no_grad():
            outputs = model(token_ids)
            logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), token_ids.view(-1))

        total_loss += loss.item()
        total_tokens += token_ids.size(1)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


test_qa, test_sum = load_dataset()
perplexity = calculate_perplexity(tokenizer, model, test_qa)
print("QA Perplexity:", perplexity)

perplexity = calculate_perplexity(tokenizer, model, test_sum)
print("SUM Perplexity:", perplexity)

qa_results = []
for sample in tqdm(test_qa):
    qa_results.append(inference(sample, task='qa'))

with open(f'results/qa_checkpoint-40.json', 'w') as f:
    json.dump(qa_results,f)

sum_results = []
for sample in tqdm(test_sum):
    sum_results.append(inference(sample, task='sum'))

with open(f'results/sum_checkpoint-40.json', 'w') as f:
    json.dump(qa_results,f)
