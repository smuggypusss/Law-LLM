import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 1. CONFIGURATION ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi3-law-adapter" # The directory where your adapter was saved

# --- 2. DEFINE QUANTIZATION CONFIG ---
# We need to use the same 4-bit configuration as in training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --- 3. LOAD THE TOKENIZER AND QUANTIZED BASE MODEL ---
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    # Pass the quantization config here
    quantization_config=bnb_config
)

# --- 4. LOAD THE LoRA ADAPTER ---
# Now, PEFT will correctly attach the adapter to the quantized base model
model = PeftModel.from_pretrained(model, adapter_path)

# --- 5. PREPARE A PROMPT ---
judgment_text = """
The appellant, a public servant, was dismissed from service following a departmental inquiry. 
He challenged the dismissal on the grounds that the inquiry was not conducted in accordance with the principles of natural justice, 
as he was not given an opportunity to cross-examine key witnesses. The High Court dismissed his petition. He has now appealed to the Supreme Court.
"""

prompt = (
    f"<|im_start|>system\n"
    f"You are a specialized Indian legal AI assistant. Your task is to meticulously analyze the provided Supreme Court of India judgment and provide a concise summary of its key points, reasoning, and final decision.\n"
    f"<|im_end|>\n"
    f"<|im_start|>user\n"
    f"Please analyze and summarize the following judgment:\n\n"
    f"JUDGMENT TEXT:\n{judgment_text}\n"
    f"<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

# --- 6. GENERATE A RESPONSE ---
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")

outputs = model.generate(**inputs, max_length=512)
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("--- MODEL RESPONSE ---")
print(response_text)