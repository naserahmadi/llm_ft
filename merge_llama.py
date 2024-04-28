import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model_folder = "llama2-7b-chat-ft"
base_model_id = "NousResearch/Llama-2-7b-chat-hf"
#base_model_id = "NousResearch/Meta-Llama-3-8B-Instruct"



base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

model = AutoPeftModelForCausalLM.from_pretrained(
    base_model,
    model_folder,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

merged_model = model.merge_and_unload()

output_folder = 'merged-llama2-7b-chat-ft'

# save the merged model and the tokenizer
merged_model.save_pretrained(output_folder, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(model_folder)
tokenizer.save_pretrained(output_folder)
