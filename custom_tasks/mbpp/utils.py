import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import evaluate as hf_evaluate

import os
import pandas as pd
from datasets import Dataset

pass_at_k = hf_evaluate.load("code_eval")

# run simple test to check code execution is enabled before model generation
test_cases = ["assert add(2, 3)==5"]
candidates = [["def add(a,b): return a*b"]]
results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])

def process_docs(dataset: Dataset) -> Dataset:
    """Process MBPP dataset into the required format.
    
    Args:
        dataset: Raw MBPP dataset from HuggingFace datasets
        
    Returns:
        Dataset: Processed dataset with formatted prompts
    """
    def format_prompt(example):
        # Get the task description and test cases
        task_id = example['task_id']
        text = example['text']
        test_list = example['test_list']
        
        # Start building the prompt
        prompt = f"\n{text}\nTest examples:"
        
        # Special handling for task 493 which has very long test examples
        if task_id == 493:
            prompt += "\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in test_list:
                prompt += f"\n{test_example}"
                
        # Replace 4 spaces with tabs for consistent formatting
        prompt = prompt.replace('    ', '\t')
        prompt = generate_code_task_prompt(prompt)
        # Update the example with the formatted prompt
        example['prompt'] = prompt
        return example
    
    # Apply the formatting to each example in the dataset
    processed_dataset = dataset.map(format_prompt)
    
    return processed_dataset


def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION

    
def pass_at_1(references, predictions):
    #temp = [extract_answer_number(prediction) for prediction in predictions]
    #print(temp)
    return pass_at_k.compute(
        references=references,
        predictions=[[extract_answer_number(prediction) for prediction in predictions]],
        k=[1],
    )[0]["pass@1"]

def extract_answer_number(completion: str) -> str:
    """
    Extract the code answer from completion text by handling various formats.
    
    Args:
        completion: Raw completion text containing code
        
    Returns:
        str: Cleaned code answer with consistent formatting
    """
    completion = completion.split("### Response:")[-1]
    completion = completion.replace('\t', '    ')
    completion = completion.strip()
    
    # Handle code blocks with ```python
    if '```python' in completion:
        def_line = completion.index('```python')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('\n```')
            completion = completion[:next_line].strip()
        except ValueError:
            print("wrong completion")

    # Remove main block
    if '__name__ == "__main__"' in completion:
        try:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
        except ValueError:
            print("wrong completion")

    # Remove example usage
    if "# Example usage" in completion:
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()

    # Remove test examples
    if "# Test examples" in completion:
        next_line = completion.index('# Test examples')
        completion = completion[:next_line].strip()

    # Handle code-alpaca style outputs
    if "The solution is:" in completion:
        def_line = completion.index("The solution is:")
        completion = completion[def_line:].strip()
        completion = completion.replace('The solution is:', '')
        try:
            next_line = completion.index('\n\nThe answer is:')
            completion = completion[:next_line].strip()
        except ValueError:
            completion = completion.strip()
            print("maybe wrong completion")

    if "The answer is:" in completion:
        def_line = completion.index("The answer is:")
        completion = completion[def_line:].strip()
        completion = completion.replace('The answer is:', '')
        try:
            next_line = completion.index('\n\nThe answer is:')
            completion = completion[:next_line].strip()
        except ValueError:
            completion = completion.strip()
            print("maybe wrong completion")

    return completion

'''
class GetCode(Filter):
    """ """
    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            filtered = extract_answer_number(r)
            if filtered == None:
                filtered_resps.append("[invalid]")
            else:
                filtered_resps.append(filtered)

        return filtered_resps
'''

def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]