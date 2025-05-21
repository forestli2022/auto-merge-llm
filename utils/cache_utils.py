import os


def set_cache_dir(cache_dir_path):
    if not os.path.exists(cache_dir_path):
        os.makedirs(cache_dir_path)
        print(f"Cache directory created at: {cache_dir_path}")
    else:
        print(f"Cache directory already exists at: {cache_dir_path}")
    
    os.environ['TRANSFORMERS_CACHE'] = cache_dir_path 
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir_path, 'datasets')
    os.environ['HF_HOME'] = cache_dir_path 
    
    print(f"Transformers cache directory set to: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"Datasets cache directory set to: {os.environ['HF_DATASETS_CACHE']}")
    print(f"Hugging Face home directory set to: {os.environ['HF_HOME']}")
    
