from datasets import load_dataset

dataset = load_dataset("LLukas22/nq-simplified")

print(dataset)

# Inspect the first available split
split_name = next(iter(dataset.keys()))
example = dataset[split_name][0]

print("Question:", example["question"])
print("Context:", example["context"][:500])
print("Answers:", example["answers"])
