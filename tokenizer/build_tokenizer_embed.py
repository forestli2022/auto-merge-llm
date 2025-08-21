import json

import torch.nn.functional as F
import traceback
from utils import logger


def align_tokenizers_and_embeddings(
    pretrained_model, pretrained_tokenizer,
    pretrained_config, finetuned_models,
    finetuned_tokenizers, finetuned_configs
):
    pretrained_vocab_size = pretrained_config.vocab_size
    try:
        # examine the pretrained tokenizer
        models_vocab_size = [pretrained_vocab_size]
        # logger.info(
        #     "Vocab size of pretrained model is %d .",
        #     pretrained_vocab_size
        # )
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
            # logger.info(
            #     "Vocab size of index %d finetuned model is %d.",
            #     index,
            #     finetuned_vocab_size
            # )
            # logger.info(f"Added pad tokens of index {index} finetuned model is {finetuned_added_pad_tokens}.")
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            finetuned_added_vocab_size = (
                finetuned_vocab_size - pretrained_vocab_size
            )
            if (finetuned_added_vocab_size < len(finetuned_added_pad_tokens)):
                logger.warning(f"Vocab size in index {index} finetuned model's config mismatches (less than) number of added tokens.")
                logger.warning(f"Before removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                for _ in range(len(finetuned_added_pad_tokens) - finetuned_added_vocab_size):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning("Remove pad token %s.", removed_pad_token)
                    assert removed_pad_token["content"] in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(f"After removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
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
        # logger.info(f"All added pad tokens of finetuned models are {added_pad_tokens_set}.")

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
                    # logger.info(f"Skip added pad token {added_pad_token} of index {index} model since its original added pad tokens and token embeddings are matched.")
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
            
            # logger.info(f"Aligned index {index} model: input token embedding shape {model.get_input_embeddings().weight.shape}, "
            #             f"output token embedding shape {model.get_output_embeddings().weight.shape}, "
            #             f"tokenizer added tokens {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}.")
            aligned_models_vocab_size_set.add(
                model.model.embed_tokens.weight.shape
            )
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.error(traceback.print_exc())
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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer, model,
    pretrained_vocab_size
):
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


def align_tokenizers_and_embeddings_v1(
    pretrained_model_embed_list,
    based_model_tokenizer_config,
    finetuned_models_embeds_list,
    merging_models_tokenizer_config
):
    input_aligned_embeds = []
    output_aligned_embeds = []
    pretrained_model_input_embed, pretrained_model_output_embed = pretrained_model_embed_list[0], pretrained_model_embed_list[1]
    finetuned_model_input_dict, finetuned_model_output_dict = finetuned_models_embeds_list[0], finetuned_models_embeds_list[1]
    
    if based_model_tokenizer_config:
        pretrained_tokenizer = based_model_tokenizer_config.get("tokenizer")
        pretrained_config = based_model_tokenizer_config.get("config")
    else:
        pretrained_tokenizer = None
        pretrained_config = None
    
    # logger.info("pretrained tokenizer") 
    # logger.info(pretrained_tokenizer)
    # logger.info(pretrained_config)
    
    
     
    finetuned_tokenizers = []
    finetuned_configs = []
    finetuned_model_input_embeds = []
    finetuned_model_output_embeds = []
    model_name_list = ["base"]
    for model_name, model_config in merging_models_tokenizer_config.items():
        tokenizer = model_config.get("tokenizer")
        config = model_config.get("config")
        # logger.info("finetuned tokenizer") 
        # logger.info(model_name)
        # logger.info(tokenizer)
        # logger.info(config)
        
        if tokenizer and config:
            finetuned_tokenizers.append(tokenizer)
            finetuned_configs.append(config)
            finetuned_model_input_embeds.append(finetuned_model_input_dict[model_name])
            finetuned_model_output_embeds.append(finetuned_model_output_dict[model_name])
        model_name_list.append(model_name)
    
    
    # if no pretrain config, then pretrain config should choose the smallest vocab_size one 
    if pretrained_model_input_embed is None:
        # logger.info("pretrained model input embed is none")
        min_index = min(range(len(finetuned_configs)), key=lambda i: finetuned_configs[i].vocab_size)

        pretrained_model_input_embed = finetuned_model_input_embeds[min_index]
        pretrained_tokenizer = finetuned_tokenizers[min_index]
        pretrained_config = finetuned_configs[min_index]
        pretrained_model_output_embed = finetuned_model_output_embeds[min_index]
    
    pretrained_vocab_size = pretrained_config.vocab_size

    try:
        # examine the pretrained tokenizer
        models_vocab_size = [pretrained_vocab_size]
        # logger.info(
        #     "Vocab size of pretrained model is %d.",
        #     pretrained_vocab_size
        # )
        pretrained_token_dict = json.loads(pretrained_tokenizer._tokenizer.to_str())
        pretrained_added_pad_tokens = [
            token_dict
            for token_dict in pretrained_token_dict["added_tokens"]
            if token_dict["id"] >= pretrained_vocab_size
        ]
        # logger.info(f"pretrained added pad tokens are {pretrained_added_pad_tokens}")
        assert pretrained_added_pad_tokens == []
        models_added_pad_tokens_list = [(True, pretrained_added_pad_tokens)]

        # append the added pad token of finetuned tokenizers into a set
        added_pad_tokens_set = set()
        for index, (finetuned_tokenizer, finetuned_config) in enumerate(
            zip(finetuned_tokenizers, finetuned_configs)
        ):
            finetuned_vocab_size = finetuned_config.vocab_size
            models_vocab_size.append(finetuned_vocab_size)
            finetuned_token_dict = json.loads(finetuned_tokenizer._tokenizer.to_str())
            finetuned_added_pad_tokens = [
                token_dict 
                for token_dict in finetuned_token_dict["added_tokens"] 
                if token_dict["id"] >= pretrained_vocab_size
            ]
            # logger.info(f"Vocab size of index {index} finetuned model is {finetuned_vocab_size}.")
            # logger.info(f"Added pad tokens of index {index} finetuned model is {finetuned_added_pad_tokens}.")
           
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            finetuned_added_vocab_size = (
                finetuned_vocab_size - pretrained_vocab_size
            )
            if finetuned_added_vocab_size < len(finetuned_added_pad_tokens):
                logger.warning(f"Vocab size in index {index} finetuned model's config mismatches (less than) number of added tokens.")
                logger.warning(f"Before removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                
                for _ in range(len(finetuned_added_pad_tokens) - finetuned_added_vocab_size):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning("Remove pad token %s.", removed_pad_token)
                    assert (
                        removed_pad_token["content"] 
                        in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                    )
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(f"After removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
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
        # logger.info(f"All added pad tokens of finetuned models are {added_pad_tokens_set}.")

        # align the tokenizers
        aligned_models_vocab_size_set = set()
        for index, (input_embed, output_embed, tokenizer, model_vocab_size) in enumerate(
            zip( 
                [pretrained_model_input_embed] + finetuned_model_input_embeds, 
                [pretrained_model_output_embed] + finetuned_model_output_embeds, 
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
                    # logger.info(f"Skip added pad token {added_pad_token} of index {index} model since its original added pad tokens and token embeddings are matched.")
                    continue
                num_new_tokens = tokenizer.add_special_tokens(
                    {"pad_token": added_pad_token}
                )
                if num_new_tokens > 0:
                    assert num_new_tokens == 1
                    model_vocab_size = model_vocab_size + num_new_tokens
                    input_embed = F.pad(input_embed, (0, 0, 0, num_new_tokens), "constant", 0)
                    embeddings_avg = input_embed[:-num_new_tokens].mean(dim=0, keepdim=True)
                    input_embed[-num_new_tokens:] = embeddings_avg
                    
                    output_embed = F.pad(output_embed, (0, 0, 0, num_new_tokens), "constant", 0)
                    embeddings_avg = output_embed[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embed[-num_new_tokens:] = embeddings_avg
                    
            input_aligned_embeds.append(input_embed)
            output_aligned_embeds.append(output_embed)
            
            # logger.info(f"Aligned index {index} model: input token embedding shape {input_embed.shape}, "
                        # f"output token embedding shape {output_embed.shape}, "
                        # f"tokenizer added tokens {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}.")
            
            aligned_models_vocab_size_set.add(input_embed.shape)
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.info(traceback.print_exc())
        logger.error(e)
        logger.warning(
            "Unable to align tokenizers by default function,"
            "using alternative smart_tokenizer_and_embedding_resize function."
        )
        for input_embed, output_embed, tokenizer in zip(
            [pretrained_model_input_embed] + finetuned_model_input_embeds, 
            [pretrained_model_output_embed] + finetuned_model_output_embeds, 
            [pretrained_tokenizer] + finetuned_tokenizers
        ):
            aligned_input_embed, aligned_output_embed = smart_tokenizer_and_embedding_resize_v1(
                special_tokens_dict={"pad_token": "<special_pad>"},
                tokenizer=tokenizer, 
                input_embed=input_embed,
                output_embed=output_embed,
                pretrained_vocab_size=pretrained_vocab_size
            )
            input_aligned_embeds.append(aligned_input_embed)
            output_aligned_embeds.append(aligned_output_embed)
    
    result_dict = {
        model_name: {
            "input_aligned_embed": input_embed,
            "output_aligned_embed": output_embed
        }
        for model_name, input_embed, output_embed in zip(model_name_list, input_aligned_embeds, output_aligned_embeds)
    } 
    return result_dict


def smart_tokenizer_and_embedding_resize_v1(
    special_tokens_dict, 
    tokenizer, 
    input_embed, 
    output_embed, 
    pretrained_vocab_size
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        input_embed = F.pad(input_embed, (0, 0, 0, num_new_tokens), "constant", 0)
        embeddings_avg = input_embed[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embed[-num_new_tokens:] = embeddings_avg
        
        output_embed = F.pad(output_embed, (0, 0, 0, num_new_tokens), "constant", 0)
        embeddings_avg = output_embed[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embed[-num_new_tokens:] = embeddings_avg
    return input_embed, output_embed