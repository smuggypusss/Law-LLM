from datasets import load_dataset
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
        f"SUMMARY:\n{example['summary']}"  # The model learns to generate this part
        f"<|im_end|>"
    )
    return {"texts":prompt}

small_dataset=load_dataset("percins/IN-ABS",split="train[:1000]")
formatted_dataset=small_dataset.map(legal_assistant_prompt)

print(formatted_dataset[0]['texts'])