import os
import json

BENCHMARKS = ["boolq", "commonsense_qa", "hellaswag", "mmlu", "piqa", "race", "wsc"]

def main(output_path, single_eval):
    # Check if output path exists
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output path {output_path} does not exist.")

    averages = {}

    if single_eval:
        # Expect a single evaluation result directly under output_path
        result_path = os.path.join(output_path, "evaluation result.json")
        if not os.path.exists(result_path):
            raise RuntimeError(f"Expected single evaluation result at {result_path}, but file not found.")

        with open(result_path, "r") as f:
            data = json.load(f)

        # Verify all benchmarks exist
        if not all(key in data for key in BENCHMARKS):
            raise RuntimeError(f"Not all benchmark results found in {result_path}.")

        total = 0
        total_star = 0
        unused_data_count = 0
        for key in BENCHMARKS:
            value = data[key]
            score_key = "score" if "score" in value else "acc,none"
            total += value[score_key]
            if not any(sub in key for sub in ["mmlu", "wsc", "piqa", "commonsense_qa"]):
                total_star += value[score_key]
                unused_data_count += 1

        averages["single"] = {
            "avg": total / len(BENCHMARKS),
            "avg*": total_star / (unused_data_count if unused_data_count > 0 else len(BENCHMARKS))
        }

    else:
        # Multiple subdirectories case
        indices = [d for d in os.listdir(output_path) if d.isdigit()]

        for idx in indices:
            index_path = os.path.join(output_path, idx)
            if not os.path.isdir(index_path):
                raise RuntimeError(f"Result from index {idx} missing!")

            result_path = os.path.join(index_path, f"evaluation result {idx}.json")
            if not os.path.exists(result_path):
                continue

            with open(result_path, "r") as f:
                data = json.load(f)

            # Check each benchmark key is present
            if not all(key in data for key in BENCHMARKS):
                continue

            total = 0
            total_star = 0
            unused_data_count = 0
            for key in BENCHMARKS:
                value = data[key]
                score_key = "score" if "score" in value else "acc,none"
                total += value[score_key]
                if not any(sub in key for sub in ["mmlu", "wsc", "piqa", "commonsense_qa"]):
                    total_star += value[score_key]
                    unused_data_count += 1

            averages[idx] = {
                "avg": total / len(BENCHMARKS),
                "avg*": total_star / (unused_data_count if unused_data_count > 0 else len(BENCHMARKS))
            }

        # Sort by index keys
        averages = dict(sorted(averages.items(), key=lambda item: int(item[0])))

    # Write results
    out_file = os.path.join(output_path, "averages.json")
    with open(out_file, "w") as f:
        print(f"Writing averages to {out_file}")
        json.dump(averages, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate average evaluation results.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory containing evaluation results.")
    parser.add_argument("--single_eval", action="store_true", help="If set, only calculate the average for a single evaluation result.")
    args = parser.parse_args()
    
    main(args.output_path, args.single_eval)
