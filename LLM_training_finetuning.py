import torch
import wandb
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,BitsAndBytesConfig
from peft import LoraConfig,TaskType,get_peft_model
from trl import SFTTrainer,SFTConfig
from datasets import load_dataset
import pandas as pd


MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
DATASET_ID="percins/IN-ABS"
WANDB_PROJECT="Law_LLM"
OUTPUT_DIR="phi3-law-adapter"



def legal_assistant_prompt(example):
    prompt=(
        f"<|im_start|>system\n"
        f"You are a specialized Indian legal AI assistant. Your task is to meticulously analyze the provided Supreme Court of India judgment and provide a concise summary of its key points, reasoning, and final decision.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Please analyze and summarize the following judgment:\n\n"
        f"JUDGMENT TEXT:\n{example['text']}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"SUMMARY:\n{example['summary']}" 
        f"<|im_end|>"
    )
    return {"texts":prompt}

small_dataset=load_dataset("percins/IN-ABS",split="train[:1000]")
formatted_dataset=small_dataset.map(legal_assistant_prompt)
#loading the dataset
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

#Quantization and model loading
quant_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
torch.cuda.empty_cache()

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
)


# PEFT and LoRA config
lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules='all-linear',
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,

)

model = get_peft_model(model, lora_config)



# Training Arguments

sft_config=SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    max_steps=500,
    fp16=False,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
    run_name=f"run-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
    dataset_text_field="text",
    max_seq_length=512,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False},

)



#Initializing trainer
trainer=SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    #peft_config=lora_config,
    tokenizer=tokenizer,
    args=sft_config,

)
torch.cuda.empty_cache()
trainer.train()
trainer.save_model(OUTPUT_DIR)