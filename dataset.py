from datasets import load_dataset
import pandas as pd

dataset = load_dataset("percins/IN-ABS",split="train")

print(dataset[0])


example_text=next(iter(dataset))

print(example_text.keys())
#print(f"Judgement \n{example_text['text'][:500]}")
#print(f"Summary \n{example_text['summary']}")