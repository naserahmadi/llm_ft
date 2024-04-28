from datasets import load_dataset
import json

sum_dataset = load_dataset("samsum")
qa_dataset = load_dataset('lucadiliello/triviaqa')


def prompt_formatter_summ(sample):
	return f"""<s>### Instruction:
You are a helpful, respectful and honest assistant. \
Your task is to summarize the following dialogue. \
Your answer should be based on the provided dialogue only.

### Dialogue:
{sample['dialogue']}

### Summary:
{sample['summary']} </s>"""


def prompt_formatter_qa(sample):
	return f"""<s>### Instruction:
You are a helpful, respectful and honest assistant. \
Your task is to answer the following question. \
Your answer should be based on the provided context only.

### Question:
{sample['question']}

### Context:
{sample['context']}


### Answer:
{sample['answers'][0]} </s>"""



## Summarization dataset
train_sum, val_sum, test_sum = [],[],[]

for i in range(0, len(sum_dataset['train'])):
    sample = sum_dataset['train'][i]
    train_sum.append({'instruction' : prompt_formatter_summ(sample)})

for i in range(0, len(sum_dataset['test'])):
    sample = sum_dataset['test'][i]
    test_sum.append({'instruction': prompt_formatter_summ(sample)})

for i in range(0, len(sum_dataset['validation'])):
    sample = sum_dataset['validation'][i]
    val_sum.append({'instruction':prompt_formatter_summ(sample)})


## qa dataset
train_qa, val_qa, test_qa = [],[],[]

for i in range(0, len(sum_dataset['train'])):
    sample = qa_dataset['train'][i]
    train_qa.append({'instruction': prompt_formatter_qa(sample)})

for i in range(0, len(sum_dataset['test'])):
    sample = qa_dataset['validation'][i]
    test_qa.append({'instruction':prompt_formatter_qa(sample)})

for i in range(len(sum_dataset['test']), len(sum_dataset['validation']) + len(sum_dataset['test'])):
    sample = qa_dataset['validation'][i]
    val_qa.append({'instruction':prompt_formatter_qa(sample)})


with open('data/train.json', 'w') as f:
    json.dump(train_qa+train_sum,f)

with open('data/val.json', 'w') as f:
    json.dump(val_qa+val_sum,f)

with open('data/test_sum.json', 'w') as f:
    json.dump(test_sum,f)

with open('data/test_qa.json', 'w') as f:
    json.dump(test_qa,f)