import json
import torch
from random import shuffle
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


model_id = "NousResearch/Llama-2-7b-chat-hf"
#model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

args = TrainingArguments(
    output_dir="llama2-7b-chat-ft",
    num_train_epochs=1,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=2,
    logging_steps=4,
    save_strategy="steps",
    save_steps= 100,
    save_total_limit=5,
    learning_rate=2e-4,
    optim="paged_adamw_32bit",
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,
)


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules= [
        'lm_head',
        'down_proj',
        'o_proj',
        'k_proj',
        'q_proj',
        'gate_proj',
        'up_proj',
        'v_proj'],
    bias="none",
    task_type="CAUSAL_LM", 
)


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",)
    
    model = prepare_model_for_kbit_training(model)

    return model


def load_dataset():
    with open('data/train.json', 'r') as f:
        train_dataset = json.load(f)

    with open('data/test.json', 'r') as f:
        test_dataset = json.load(f)

    with open('data/val.json', 'r') as f:
        val_dataset = json.load(f)

    shuffle(train_dataset)
    shuffle(test_dataset)
    shuffle(val_dataset)

    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    model = load_model()

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, val_dataset, test_dataset = load_dataset()


    trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    dataset_text_field='instruction'
    )

    trainer.train()

    trainer.save_model()
