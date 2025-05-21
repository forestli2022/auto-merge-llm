from datasets import load_dataset, Dataset, Features, Value, Sequence
import random
import json
import os

def create_new_validation_set(
    data_name="mbpp_new_validation", 
    output_dir="/home/gsu/workspace/mergebase/mo_custom_tasks/mbpp",
    seed=42
):
    """
    Create a new validation dataset in HuggingFace datasets format.
    """
    # Set random seed
    random.seed(seed)
    
    # Create full path with data_name
    dataset_dir = os.path.join(output_dir, data_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load original MBPP dataset
    dataset = load_dataset("google-research-datasets/mbpp")
    
    # Convert to list and preserve all fields
    train_list = [dict(item) for item in dataset['train']]
    valid_list = [dict(item) for item in dataset['validation']]
    test_list = [dict(item) for item in dataset['prompt']]
    
    # Sample from train
    sampled_train = random.sample(train_list, 200)
    
    # Combine all data for new validation set
    new_validation = sampled_train + valid_list + test_list
    
    # Save data in JSONL format
    validation_path = os.path.join(dataset_dir, "validation.jsonl")
    with open(validation_path, 'w') as f:
        for example in new_validation:
            f.write(json.dumps(example) + '\n')

    # Create dataset info file
    dataset_info = {
        "builder_name": "json",
        "config_name": "default",
        "version": "1.0.0",
        "splits": {
            "validation": {
                "name": "validation",
                "num_bytes": os.path.getsize(validation_path),
                "num_examples": len(new_validation),
                "dataset_name": data_name
            }
        },
        "download_checksums": {},
        "download_size": 0,
        "dataset_size": os.path.getsize(validation_path),
        "size_in_bytes": os.path.getsize(validation_path)
    }
    
    # Save dataset info
    with open(os.path.join(dataset_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Create dataset dict file
    data_files = {
        "validation": "validation.jsonl"
    }
    
    with open(os.path.join(dataset_dir, "dataset_dict.json"), "w") as f:
        json.dump(data_files, f, indent=2)
        
    print(f"Created dataset '{data_name}' with {len(new_validation)} examples")
    print(f"Dataset saved to '{dataset_dir}'")
    
    print("\nFirst example structure:")
    print(json.dumps(new_validation[0], indent=2))

if __name__ == "__main__":
    # Create the dataset
    create_new_validation_set()
    
    # Test loading
    dataset = load_dataset(
        path="/home/gsu/workspace/mergebase/mo_custom_tasks/mbpp/mbpp_new_validation",
        name="default"
    )
    print("\nLoaded dataset structure:")
    print(dataset)
    print("\nVerifying first example:")
    print(dataset['validation'][0])
    
    
    