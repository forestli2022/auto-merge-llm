import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import evaluate as hf_evaluate


pass_at_k = hf_evaluate.load("code_eval")

# run simple test to check code execution is enabled before model generation
test_cases = ["assert add(2, 3)==5"]
candidates = [["def add(a,b): return a*b"]]
results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])


def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION

def process_docs(dataset):
    def format_prompt(example):
        prompt = example['prompt'].replace('    ', '\t')
        prompt = generate_code_task_prompt(prompt)
        # Update the example with the formatted prompt
        example['prompt'] = prompt
        return example
        
    processed_dataset = dataset.map(format_prompt)
    return processed_dataset


def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=[[prediction for prediction in predictions]],
        k=[1],
    )[0]["pass@1"]


def extract_prediction(completion):
    completion_seq = gen_seq.split("### Response:")[-1]
    completion_seq = completion_seq.replace('\t', '    ')
    completion = completion.replace("\r", "")
    completion = completion.strip()
    if '```python' in completion:
        print("completion matches ```python")
        def_line = completion.index('```python')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('```')
            completion = completion[:next_line].strip()
        except:
            print("wrong completion")
    if "__name__ == \"__main__\"" in completion:
        print("completion matches __name__ == \"__main__\"")
        try:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
        except:
            print("wrong completion")
    if "# Example usage" in completion:
        print("completion matches # Example usage")
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()
    # the following codes are used to deal with the outputs of code-alpaca
    if "The solution is:" in completion:
        print("completion matches The solution is:")
        def_line = completion.index("The solution is:")
        completion = completion[def_line:].strip()
        completion = completion.replace('The solution is:', '')
        try:
            next_line = completion.index('\n\nThe answer is:')
            completion = completion[:next_line].strip()
        except:
            completion = completion.strip()
            print("maybe wrong completion")
    if "The answer is:" in completion:
        print("completion matches The answer is:")
        def_line = completion.index("The answer is:")
        completion = completion[def_line:].strip()
        completion = completion.replace('The answer is:', '')
        try:
            next_line = completion.index('\n\nThe answer is:')
            completion = completion[:next_line].strip()
        except:
            completion = completion.strip()
            print("maybe wrong completion")
    return completion
    


def build_references(doc):
    return doc["test"] + "\n" + f"check({doc['entry_point']})"


def build_predictions(resps, docs):
    preds = []
    for resp, doc in zip(resps, docs):
        pred = [doc["prompt"] + extract_prediction(r) for r in resp]
        preds.append(pred)

    return preds