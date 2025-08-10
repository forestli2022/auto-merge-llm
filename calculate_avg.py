import os
import json

BENCHMARKS = ["boolq", "commonsense_qa", "hellaswag", "mmlu", "piqa", "race", "wsc"]

def main(output_path):
    # Calculate the average of each configuration's evaluation results
    # first search for directory named with number stored under the output path
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output path {output_path} does not exist.")

    indices = [d for d in os.listdir(output_path) if d.isdigit()]

    averages = {}
    for idx in indices:
        index_path = os.path.join(output_path, idx)
        if not os.path.isdir(index_path):
            raise RuntimeError(f"Result from index {idx} missing!")

        # Read the evaluation results from evaluation result {i}.json
        result_path = os.path.join(index_path, f"evaluation result {idx}.json")

        unused_data_count = 0
        with open(result_path, "r") as f:
            data = json.load(f) # data is a dictionary
            total = 0
            total_star = 0
            for key in BENCHMARKS:
                value = data[key]
                score_key = "score" if "score" in value else "acc,none"
                total += value[score_key]
                # If key contains ["mmlu", "wsc", "piqa", "commonsense_qa"], do not add it to avg_star
                if not any(sub in key for sub in ["mmlu", "wsc", "piqa", "commonsense_qa"]):
                    unused_data_count += 1
                    total_star += value[score_key]
        
        # Calculate averages
        averages[idx] = {}
        averages[idx]["avg"] = total / len(BENCHMARKS)
        averages[idx]["avg*"] = total_star / (unused_data_count if unused_data_count > 0 else len(BENCHMARKS))
        
    # Sort the dictionary with indexes as keys
    averages = dict(sorted(averages.items(), key=lambda item: int(item[0])))

    # Store the averages in a JSON file
    with open(os.path.join(output_path, "averages.json"), "w") as f:
        print(f"Writing averages to {os.path.join(output_path, 'averages.json')}")
        json.dump(averages, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate average evaluation results.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory containing evaluation results.")
    args = parser.parse_args()
    
    main(args.output_path)