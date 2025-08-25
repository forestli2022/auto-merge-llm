import argparse
from utils import logger, set_cache_dir, seed_everything, load_and_validate_config

def get_merge_strategy(strategy_name, config):
    from strategies import strategy_classes
    strategy_class = strategy_classes.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(config)  


def main(config, weight_path, output_path=None):
    logger.info("start evaluating")
    set_cache_dir(config.get("cache_dir"))
    seed_everything(config.get("random_seed"))
    selected_strategy = config.get('strategy')

    merge_strategy_instance = get_merge_strategy(selected_strategy, config)
    merge_strategy_instance.evaluate(weight_path, output_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Strategy Application")
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--weight_path', type=str, help='Path to the result weight/config of search', required=True)
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the evaluation results')
    args = parser.parse_args()
    config = load_and_validate_config(args.config) 
    weight_path = args.weight_path
    output_path = args.output_path
    main(config, weight_path, output_path)
