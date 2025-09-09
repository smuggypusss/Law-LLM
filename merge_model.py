import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "C:/Users/Asus/OneDrive/Desktop/LLMs/phi3-law-adapter"
output_dir = "./phi3-indian-law-merged"


print("Base Model Loaded")
base_model=AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Tokenizer loading")
tokenizer=AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)

print("Loading peft model")
model=PeftModel.from_pretrained(
    base_model,
    adapter_path,
)


print("Merge PEFT and Pretrained model")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
