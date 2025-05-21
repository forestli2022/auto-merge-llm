import re
import os
import json
import random

import numpy as np
import torch
from transformers import TrainerState
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, Polygon


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["DATA_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_model_storage_path(model_name):
    if os.path.isabs(model_name) or os.path.exists(model_name):
        return model_name
    CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE')
    model_folder_path = os.path.join(CACHE_DIR, 'models--' + model_name.replace('/', '--'))
    
    if not os.path.exists(model_folder_path):
        raise FileNotFoundError(f"model folder path doesn't exist: {model_folder_path}")
    
    snapshots_dir = os.path.join(model_folder_path, 'snapshots')
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"can not find snapshots dir: {snapshots_dir}")
    
    snapshot_subdirs = os.listdir(snapshots_dir)
    if not snapshot_subdirs:
        raise FileNotFoundError("no sub dir in ")
    
    snapshot_dir = os.path.join(snapshots_dir, snapshot_subdirs[0])
    return snapshot_dir


def save_state_and_model_for_hf_trainer(trainer):
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {
            key: value.cpu()
            for key, value
            in trainer.model.state_dict().items()}
        
        trainer._save(
            trainer.args.output_dir, 
            state_dict=cpu_state_dict
        )

def load_state_and_model_for_hf_trainer(model, load_model_dir,
                                        map_location=None):
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(
        os.path.join(load_model_dir, "pytorch_model.bin"),
        map_location=map_location)
    )
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(
        os.path.join(load_model_dir, "trainer_state.json")
    )
    return model, trainer_state


def get_param_names_to_merge(input_param_names, exclude_param_names_regex):
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([
            re.match(exclude_pattern, param_name)
            for exclude_pattern
            in exclude_param_names_regex
        ])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model, include_module_types):
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any(
            [isinstance(module, include_module_type) 
             for include_module_type 
             in include_module_types]
        )
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def align_tokenizers_and_embeddings(pretrained_model, pretrained_tokenizer,
                                    pretrained_config, finetuned_models,
                                    finetuned_tokenizers, finetuned_configs,
                                    logger):
    pretrained_vocab_size = pretrained_config.vocab_size
    try:
        # examine the pretrained tokenizer
        models_vocab_size = [pretrained_vocab_size]
        logger.info(
            "Vocab size of pretrained model is %d .",
            pretrained_vocab_size
        )
        pretrained_token_dict = json.loads(
            pretrained_tokenizer._tokenizer.to_str()
        )
        pretrained_added_pad_tokens = [
            token_dict
            for token_dict in pretrained_token_dict["added_tokens"]
            if token_dict["id"] >= pretrained_vocab_size
        ]
        assert pretrained_added_pad_tokens == []
        models_added_pad_tokens_list = [(True, pretrained_added_pad_tokens)]

        # append the added pad token of finetuned tokenizers into a set
        added_pad_tokens_set = set()
        for index, (finetuned_tokenizer, finetuned_config) in enumerate(
            zip(finetuned_tokenizers, finetuned_configs)
        ):
            finetuned_vocab_size = finetuned_config.vocab_size
            models_vocab_size.append(finetuned_vocab_size)
            finetuned_token_dict = json.loads(
                finetuned_tokenizer._tokenizer.to_str()
            )
            finetuned_added_pad_tokens = [
                token_dict 
                for token_dict in finetuned_token_dict["added_tokens"] 
                if token_dict["id"] >= pretrained_vocab_size
            ]          
            logger.info(
                "Vocab size of index %d finetuned model is %d.",
                index,
                finetuned_vocab_size
            )
            logger.info(
                "Added pad tokens of index %d finetuned model is %s.",
                index,
                ', '.join(finetuned_added_pad_tokens)
            )
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            finetuned_added_vocab_size = (
                finetuned_vocab_size - pretrained_vocab_size
            )
            if (finetuned_added_vocab_size < len(finetuned_added_pad_tokens)):
                logger.warning(
                    "Vocab size in index %d finetuned model's config mismatches "
                    "(less than) number of added tokens.",
                    index
                )
                logger.warning(
                    "Before removing pad tokens, the added tokens are %s .",
                    {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}
                )
                for _ in range(len(finetuned_added_pad_tokens) - finetuned_added_vocab_size):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning("Remove pad token %s.", removed_pad_token)
                    assert removed_pad_token["content"] in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(
                    "After removing pad tokens, the added tokens are %s ",
                    {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}
                )
                is_matched = False
            else:
                assert finetuned_added_vocab_size == len(finetuned_added_pad_tokens)
                is_matched = True
            for token_dict in finetuned_added_pad_tokens:
                added_pad_tokens_set.add(token_dict["content"])
            models_added_pad_tokens_list.append((
                is_matched, 
                [
                    token_dict["content"] 
                    for token_dict in finetuned_added_pad_tokens
                ]
            ))
        logger.info(
            "All added pad tokens of finetuned models are %s",
            ", ".join(added_pad_tokens_set)
        )

        # align the tokenizers
        aligned_models_vocab_size_set = set()
        for index, (model, tokenizer, model_vocab_size) in enumerate(
            zip(
                [pretrained_model] + finetuned_models,
                [pretrained_tokenizer] + finetuned_tokenizers,
                models_vocab_size
            )
        ):
            is_matched = models_added_pad_tokens_list[index][0]
            model_added_pad_tokens_list = models_added_pad_tokens_list[index][1]
            for added_pad_token in added_pad_tokens_set:
                # deal with models like llama-2-13b-code-alpaca, whose finetuned_token_dict['added_tokens'] contains pad tokens and token embeddings are added,
                # but tokenizer.add_special_tokens({"pad_token": "<pad>"}) returns 1 instead of 0 (this model does not have tokenizer.json file)
                if is_matched and added_pad_token in model_added_pad_tokens_list:
                    logger.info(
                        "Skip added pad token %s of index %d model "
                        "since its original added pad tokens and token embeddings are matched.",
                        added_pad_token,
                        index
                    )
                    continue
                num_new_tokens = tokenizer.add_special_tokens({"pad_token": added_pad_token})
                if num_new_tokens > 0:
                    assert num_new_tokens == 1
                    model_vocab_size = model_vocab_size + num_new_tokens
                    model.resize_token_embeddings(new_num_tokens=model_vocab_size)

                    # shape (new_num_tokens, embed_dim)
                    input_embeddings = model.get_input_embeddings().weight.data
                    output_embeddings = model.get_output_embeddings().weight.data

                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

            logger.info(
                "Aligned index %d model: input token embedding shape %d",
                index,
                model.get_input_embeddings().weight.shape
            )           
            logger.info(
                "output token embedding shape %d",
                model.get_output_embeddings().weight.shape
            )
            logger.info(
                "tokenizer added tokens %s",
                {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}
            )
            aligned_models_vocab_size_set.add(
                model.model.embed_tokens.weight.shape
            )
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.error(e)
        logger.warning(
            "Unable to align tokenizers by default function," 
            "using alternative smart_tokenizer_and_embedding_resize function."
        )
        for model, tokenizer in zip(
            [pretrained_model] + finetuned_models, 
            [pretrained_tokenizer] + finetuned_tokenizers
        ):
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict={"pad_token": "<special_pad>"},
                tokenizer=tokenizer, 
                model=model, 
                pretrained_vocab_size=pretrained_vocab_size
            )


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                         model, pretrained_vocab_size):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(pretrained_vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = (
            input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        )
        output_embeddings_avg = (
            output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        


# Define colors based on model names to ensure color consistency
model_color_map = {
    "vanillaOVO/WizardMath-13B-V1.0": '#ff9999',        
    "WizardLMTeam/WizardLM-13B-V1.2": '#66b3ff',      
    "layoric/llama-2-13b-code-alpaca": "#ffcc99", 
    "meta-llama/Llama-2-13b-hf": "#99ff99"
}


# plotting archs (quite ugly though)
def get_figure(slices, save_path, output_scales=None):
    # Calculate total layers for dynamic height adjustment
    total_layers = sum([slice_data["sources"][0]["layer_range"][1] - slice_data["sources"][0]["layer_range"][0] for slice_data in slices])
    
    # Set up the figure with dynamic size based on the total layers and extra width for labels
    fig_height = max(15, total_layers * 1.5) 
    fig_width = 15  
    fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 

    ax.set_xlim(-2, 5)
    ax.set_ylim(0, total_layers * 1.5 + 5)

    new_layer_counter = 0
    for slice_idx, slice_data in enumerate(slices[::-1]):  # Draw slices[0] at the bottom in the correct order
        num_layers = slice_data["sources"][0]["layer_range"][1] - slice_data["sources"][0]["layer_range"][0]
        y_start = new_layer_counter * 1.5 + 1
        y_end = (new_layer_counter + num_layers - 1) * 1.5 + 2

        rect = Rectangle((1, y_start), 1, y_end - y_start, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        num_sources = len(slice_data["sources"])
        for i, source in enumerate(slice_data["sources"]):
            color = model_color_map.get(source["model"], "#cccccc")
            if num_sources == 1:
                ax.add_patch(Rectangle((1, y_start), 1, y_end - y_start, color=color, alpha=0.5))
                ax.text(1.5, (y_start + y_end) / 2, f"{source['layer_range'][0]}-{source['layer_range'][1]}", 
                        ha='center', va='center', fontsize=8, color="black")
            elif num_sources == 2:
                if i == 0:
                    ax.add_patch(Polygon(((1, y_end), (2, y_start), (1, y_start)), color=color, alpha=0.5))
                    ax.text(1.25, (y_start + y_end) / 2, f"{source['layer_range'][0]}-{source['layer_range'][1]}", 
                            ha='center', va='center', fontsize=8, color="black")
                else:
                    ax.add_patch(Polygon(((1, y_end), (2, y_end), (2, y_start)), color=color, alpha=0.5))
                    ax.text(1.75, (y_start + y_end) / 2, f"{source['layer_range'][0]}-{source['layer_range'][1]}", 
                            ha='center', va='center', fontsize=8, color="black")
            elif num_sources == 3:
                height = (y_end - y_start) / 3
                ax.add_patch(Rectangle((1, y_start + i * height), 1, height, color=color, alpha=0.5))
                ax.text(1.5, y_start + (i + 0.5) * height, f"{source['layer_range'][0]}-{source['layer_range'][1]}", 
                        ha='center', va='center', fontsize=8, color="black")

        # Dynamically handle merging methods and stagger labels to avoid overlap
        method_text = ""
        for method, params in slice_data["merging_method"].items():
            method_text += f"{method}:\n"
            for key, param_set in params.items():
                method_text += f"{key}\n"
                for param in param_set:
                    param_info = " | ".join([f"{k}: {v}" for k, v in param.items()])
                    method_text += f"{param_info}\n"

        ax.text(-1, (y_start + y_end) / 2 + (slice_idx % 2) * 0.5, method_text.strip(), 
                ha='right', va='center', fontsize=8)

        ax.text(3.5, (y_start + y_end) / 2 - (slice_idx % 2) * 0.5, 
                f"New Layers {total_layers-new_layer_counter - num_layers}-{total_layers-new_layer_counter}", 
                ha='left', va='center', fontsize=8, color="black")
        
        if output_scales != None:
            ax.text(3.5, (y_start + y_end) / 2 - (slice_idx % 2) -1.5, 
                    f"New Layers scales {output_scales[total_layers-new_layer_counter - num_layers]}-{output_scales[total_layers-new_layer_counter-1]}", 
                    ha='left', va='center', fontsize=8, color="blue")

        new_layer_counter += num_layers

    for model_name, color in model_color_map.items():
        y_legend = fig_height - 3 - list(model_color_map.keys()).index(model_name) * 2
        ax.add_patch(Rectangle((3.5, y_legend), 0.5, 1, color=color, alpha=0.5))
        ax.text(4.2, y_legend + 0.5, model_name, va='center', fontsize=8)

    ax.axis('off')
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
