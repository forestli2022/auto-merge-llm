import os
import re
import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from transformers import AutoTokenizer, AutoConfig

from loader import TensorLoader, TensorWriter
from methods import merging_methods_dict
from tokenizer import align_tokenizers_and_embeddings_v1
from utils import get_model_storage_path, logger


class MergeUtils:
    def __init__(
        self, 
        base_model, 
        merging_models, 
        merging_method,
        slices,
        model_storage_path, 
        in_memory, 
        output_scales=None, 
        device="cpu"
    ):
        
        self.base_model = base_model
        self.merging_models = merging_models
        self.slices = slices
        self.output_scales = output_scales
        self.model_storage_path = model_storage_path
        self.in_memory = in_memory
        self.device = device
        self.current_layer_offset = 0
        
        assert (self.merging_models is not None) != (self.slices is not None), \
            "Exactly one of merging_models or slices must be provided."
        
        # Initialize merging method
        if merging_method!=None:
            self.merging_method, self.merging_method_params = list(merging_method.items())[0]
            if self.merging_method not in merging_methods_dict:
                raise ValueError(
                    f"Unsupported merge method: {self.merging_method}"
                )
            merging_class = merging_methods_dict[self.merging_method]
            self.merge_instance = merging_class()
            
        # Ensure merging_method is provided when merging_models is used   
        if self.merging_models is not None:
            assert self.merging_method is not None, \
                "merging_method must be provided when merging_models is used."   
            
        # Placeholder attributes
        self.merging_model_caches = None
        self.base_model_cache = None
        self.arch_config = None
        self._aligned_tokenizer = None
        self.aligned_embeds_dict = None
        
        # Load tokenizers and configurations
        self.merging_models_tokenizer_config = (
            self._load_merging_models_tokenizer_config()
        )
        self.based_model_tokenizer_config = (
            self._load_tokenizer_and_config(self.base_model)
            if self.base_model
            else None
        )
        
        if self.base_model:
            self._output_config = self.based_model_tokenizer_config["config"]
        else:
            self._output_config = list(self.merging_models_tokenizer_config.values())[0]["config"]
  
        if not in_memory:
            self.tensor_writer = TensorWriter(self.model_storage_path)
        self._out_tensors = {}
       
    @property
    def out_tensors(self):
        """Get the output tensors."""
        return self._out_tensors
    
    @property
    def output_config(self):
        return self._output_config
    
    @property
    def aligned_tokenizer(self):
        """Get the aligned tokenizer."""
        return self._aligned_tokenizer
    
    def _load_tokenizer_and_config(self, model):
        model_data = {}

        CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE')
        try:
            temp_model_path = get_model_storage_path(model)
            model_data['tokenizer'] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
            
            model_data['config'] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=temp_model_path
            )
           
            print("Loaded from cache successfully.")     
        except OSError:
            model_data['tokenizer'] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model, 
                CACHE_DIR=CACHE_DIR
            )
            model_data['config'] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model, 
                CACHE_DIR=CACHE_DIR
            )
           
            print("Model not found in cache. Downloaded and loaded.")      
        return model_data

    def _extract_models_from_slices(self):
        model_set = set()  # Use a set to ensure uniqueness
        for cur_slice in self.slices:
            sources = cur_slice.get("sources", [])
            for source in sources:
                model = source.get('model')
                if model:  # Ensure the model key exists
                    model_set.add(model)  # Add to set for uniqueness
        return model_set

    def _load_merging_models_tokenizer_config(self):
        if not self.merging_models:
            # load from slices
            sources = self.slices[0].get("sources", []) + self.slices[-1].get("sources", [])
            return {
                source['model']: self._load_tokenizer_and_config(source['model'])
                for source in sources if 'model' in source
            }
        
        return {
            model['model']: self._load_tokenizer_and_config(model['model'])
            for model in self.merging_models
        }

    def _load_merging_model_caches(self):
        if self.merging_models:
            return {
                model: TensorLoader(
                    model_name=model,
                    lazy_unpickle=False,
                    device=self.device
                )
                for model in set([cur_model['model'] for cur_model in self.merging_models])
            }
        else:
            # Extract all unique models from slices
            model_set = self._extract_models_from_slices()
            return {
                model: TensorLoader(
                    model_name=model,
                    lazy_unpickle=False,
                    device=self.device
                )
                for model in model_set
            }
        
    def _update_output_config(self):
        if self.slices:
            num_layers = 0
            try:
                for slice_item in self.slices:
                    if "sources" in slice_item and slice_item["sources"]:
                        first_source = slice_item["sources"][0]
                        layer_range = first_source.get("layer_range")
                        if layer_range and len(layer_range) == 2:
                            num_layers += layer_range[1] - layer_range[0]
                self._output_config.update({"num_hidden_layers": int(num_layers)})
            except Exception as e:
                logger.warning(
                    "Unable to set number of layers in output config.",
                    exc_info=e,
                )
        # update vocab
        try:
            self._output_config.update({"vocab_size": int(len(self._aligned_tokenizer.get_vocab()))})
        except Exception as e:
            logger.warning(
                "Unable to set vocabulary size in output config - you may need to manually correct it.",
                exc_info=e,
            )
        if self.output_scales:
            from custom_models import CustomLlamaConfig
            config_dict = self._output_config.to_dict()
            config_dict["model_type"] = "customllama"
            self._output_config = CustomLlamaConfig.from_dict(config_dict)
            self._output_config.update({"scales": self.output_scales})
              
    def _matches_filter(self, filter_name, weight_name):
        return filter_name in weight_name 
    
    def _get_matches_weight_names(self, filter_name, match_layer=False):
        weights_names = []
        if match_layer:     
            pattern = re.compile(rf"model\.layers\.{filter_name}\..+")
            weights_names = [  
                key for key in self.arch_config.keys() 
                if pattern.search(key) is not None and "rotary_emb.inv_freq" not in key
            ]
            return weights_names
        
        pattern = re.compile(filter_name, re.IGNORECASE)
        weight_names = [
            key for key in self.arch_config.keys() if pattern.search(key)
        ]
        return weight_names
        
    def _pre_cache(self):
        self.merging_model_caches = self._load_merging_model_caches()
        self.base_model_cache = (
            TensorLoader(
                model_name=self.base_model,
                lazy_unpickle=False,
                device=self.device
            )
            if self.base_model
            else None
        )
            
        self.arch_config = (
            self.base_model_cache.tensor_paths
            if self.base_model_cache is not None
            else self.merging_model_caches[next(iter(self.merging_model_caches))].tensor_paths
        )   

    # def _get_slice_models(self, slice):
    #     sources = cur_slice['sources']
    #     return [source['model'] for source in cur_slice['sources']]
        
    def _build_tokenizer_and_embed(self):
        logger.info("start build tokenizer and embed")
        def get_single_weight(pattern):
            weights = self._get_matches_weight_names(pattern)
            assert len(weights) == 1, f"Expected single {pattern} weight, found {len(weights)}"
            return weights[0]

        input_embed_name = get_single_weight("embed")
        try:
            output_embed_name = get_single_weight("lm_head")
        except:
            output_embed_name = get_single_weight("embed_tokens")
        
        base_embeds = [
            self.base_model_cache.get_tensor(name) if self.base_model else None
            for name in [input_embed_name, output_embed_name]
        ]
       
        input_embeds = {}
        output_embeds = {}
        for model_name, _ in self.merging_models_tokenizer_config.items():
            input_embed = self.merging_model_caches[model_name].get_tensor(input_embed_name)
            output_embed = self.merging_model_caches[model_name].get_tensor(output_embed_name)
            input_embeds[model_name] = input_embed
            output_embeds[model_name] = output_embed
            
        
        self.aligned_embeds_dict = align_tokenizers_and_embeddings_v1(
            base_embeds,
            self.based_model_tokenizer_config,
            [input_embeds, output_embeds],  # embeds_to_merge
            self.merging_models_tokenizer_config
        )
      
        input_config = self._get_slice_config(self.slices[0], [input_embed_name])
        output_config = self._get_slice_config(self.slices[-1], [output_embed_name])
        self._merge_tensor(input_embed_name, self.aligned_embeds_dict['base']['input_aligned_embed'], input_config[3], input_config[0], input_config[2])
        self._merge_tensor(output_embed_name, self.aligned_embeds_dict['base']['output_aligned_embed'], output_config[3], output_config[0], output_config[2])
        self._aligned_tokenizer = self.based_model_tokenizer_config["tokenizer"] if self.based_model_tokenizer_config is not None else  list(self.merging_models_tokenizer_config.values())[0]['tokenizer']
       
         
    def _get_merge_params_by_filter(self, tensor_merge_params, weight_name):
        merged_params = {}
        for param_name, values in tensor_merge_params.items():
            default_value = None
            for value_entry in values:
                if 'filter' not in value_entry:
                    default_value = value_entry['value']
                    break

            # Find a matching entry based on the filter (if available)
            matched_value = None
            for value_entry in values:
                filter_key = value_entry.get('filter')
                if filter_key and filter_key in weight_name:
                    matched_value = value_entry['value']
                    break

            # If a match is found, use it; otherwise, use the default value
            merged_params[param_name] = matched_value if matched_value is not None else default_value
        return merged_params

    def _merge_tensor(self, weight_name, base_tensor, tensors_to_merge, tensor_merge_method, tensor_merge_params):
        tensor_merge_params = self._get_merge_params_by_filter(
            tensor_merge_params,
            weight_name
        )
        if tensor_merge_method not in merging_methods_dict.keys():
            raise ValueError(f"Unsupported merge method: {tensor_merge_method}")
        merge_class = merging_methods_dict[tensor_merge_method]
        merge_instance = merge_class()
        merged_tensor = merge_instance.merge_tensor(
            base_tensor, 
            tensors_to_merge, 
            tensor_merge_params
        )
       
        self._out_tensors[weight_name] = merged_tensor
        if not self.in_memory:
            try:
                self.tensor_writer.save_tensor(name=weight_name, tensor=merged_tensor)
            except Exception as e:
                print(f"Error saving tensor '{output_tensor.key}': {e}")
       
    def _get_slice_config(self, cur_slice, weight_names, method_key="merging_method"):
        logger.info(f"get slice config: current slice: {cur_slice}, weight_names: {weight_names}")
        sources = cur_slice['sources']
        num_sources = len(sources)
        num_weights = len(weight_names)
        if num_weights == 1 and num_sources > 1:
            weight_names = weight_names * num_sources
        if num_sources != len(weight_names):
            raise ValueError(f"Length mismatch: {num_sources} sources, {len(weight_names)} weight names.")

        tensor_merge_method, global_tensor_merge_params = list(cur_slice[method_key].items())[0]
        tensor_merge_params = []
        tensors_to_merge = []
        for source, weight_name in zip(sources, weight_names):
            model_name = source['model']
            if 'embed' in weight_name:
                current_tensor = self.aligned_embeds_dict[model_name]['input_aligned_embed']
            elif 'lm_head' in weight_name: 
                current_tensor = self.aligned_embeds_dict[model_name]['output_aligned_embed']
            else:  
                current_tensor = self.merging_model_caches[model_name].get_tensor(weight_name)
            
            tensors_to_merge.append(current_tensor)
            parameters = source.get('parameters', {})
            tensor_merge_param = {}
            for param_name, param_rules in parameters.items():
                default_value = None  # Initialize default value for this parameter
                applied = False
                for rule in param_rules:
                    if applied:
                        break
                    if 'filter' in rule:
                        # Apply the parameter if the weight name matches the filter pattern
                        if self._matches_filter(rule['filter'], weight_name):
                            default_value = rule.get('value')
                            applied = True
                    else:
                        # Store the default value in case no filter matches
                        default_value = rule.get('value')

                # If no filter matched, add the default value to the merge parameters
                if not applied and default_value is not None:
                    tensor_merge_param[param_name] = default_value
                    
            tensor_merge_params.append(tensor_merge_param)
        return tensor_merge_method, tensor_merge_params, global_tensor_merge_params,  tensors_to_merge  
             
    def _merge_layer(self, cur_slice, layer_offset):
        logger.info(f"start merge layer: current slice: {cur_slice}, current layer: {self.current_layer_offset+layer_offset}")
        sources = cur_slice["sources"]
        target_layers = [source['layer_range'][0] + layer_offset for source in sources]
        if "target" in cur_slice:
            base_target_layer = cur_slice["target"]['layer_range'][0] + layer_offset 
        else:
            base_target_layer = target_layers[0]
        #base_target_layer = target_layers[0]
        weight_names = self._get_matches_weight_names(str(base_target_layer), match_layer=True)
        for weight_name in weight_names:
            target_weight_name = weight_name.replace(f"layers.{base_target_layer}.", f"layers.{self.current_layer_offset+layer_offset}.")
            try:
                base_tensor = self.base_model_cache.get_tensor(target_weight_name)
            except:
                base_tensor = None
           
            cur_weight_names = [weight_name.replace(f"layers.{base_target_layer}.", f"layers.{target_layer}.") for target_layer in target_layers]
            tensor_merge_method, tensor_merge_params, global_tensor_merge_params, tensors_to_merge = self._get_slice_config(cur_slice, cur_weight_names)
            self._merge_tensor(target_weight_name, base_tensor, tensors_to_merge, tensor_merge_method, global_tensor_merge_params)
    
    def _merge_slice(self, cur_slice):
        logger.info(f"start merge slice {cur_slice}")
        sources = cur_slice["sources"]
        slice_lengths = [
            s['layer_range'][1] - s['layer_range'][0] for s in sources
        ]
        if not all(s == slice_lengths[0] for s in slice_lengths):
            raise RuntimeError(
                "All inputs to a slice must contain the same number of layers"
            )
            
        num_layers = slice_lengths[0]
        logger.info(f"num layer of current slice is {num_layers}")
        for idx in range(num_layers):
            self._merge_layer(
                cur_slice,
                layer_offset=idx
            )
        self.current_layer_offset += num_layers
    
    def _merge_postweights(self):
        post_norm_weights = self._get_matches_weight_names('model.norm.weight')
        assert len(post_norm_weights) == 1 
        for weight_name in [post_norm_weights[0]]:
            base_tensor = self.base_model_cache.get_tensor(weight_name) if self.base_model else None
            tensor_merge_method, tensor_merge_params, global_tensor_merge_params, tensors_to_merge = self._get_slice_config(self.slices[-1], [weight_name])
            self._merge_tensor(weight_name, base_tensor, tensors_to_merge, tensor_merge_method, global_tensor_merge_params)
    
    def _finalize_tensors(self):
        self.tensor_writer.finalize()
    
    def _save_config(self):
        self._output_config.save_pretrained(self.model_storage_path)
  
    def _save_tokenizers(self):
        self._aligned_tokenizer.save_pretrained(save_directory=self.model_storage_path)
        logger.info(f"Aligned tokenizer saved at {self.model_storage_path}.")
  
    def _save_checkpoint(self, merged_res, merging_models):
        merged_model = merged_res['merged_model']
        finetuned_tokenizers = merged_res['merged_model_tokenizers']
        based_tokenizer = merged_res['base_tokenizer']
        logger.info(f"Saving merged models at {self.model_storage_path}...")
        if not os.path.exists(self.model_storage_path):
            os.makedirs(self.model_storage_path)
        merged_model.save_pretrained(save_directory=self.model_storage_path)
        based_tokenizer.save_pretrained(save_directory=self.model_storage_path)
        
        for index, finetuned_model_name in enumerate(merging_models):
            save_tokenizer_path = os.path.join(self.model_storage_path, finetuned_model_name)
            if not os.path.exists(save_tokenizer_path):
                os.makedirs(save_tokenizer_path)
            logger.info(f"Saving each merged model's tokenizer at {save_tokenizer_path}...")
            finetuned_tokenizers[index].save_pretrained(save_directory=save_tokenizer_path)
  
    def _finalize_model(self):
        self._finalize_tensors()
        self._save_config()
        self._save_tokenizers()   
    
    def _convert_models_to_slices(self, models):
        slices = []
        num_layer = self._output_config.num_hidden_layers
        
        slice_sources = []
        for model_config in models:
            model_name = model_config.get("model")
            parameters = model_config.get("parameters", {})
            
            source = {
                "model": model_name,
                "layer_range": [0, num_layer], 
                "parameters": parameters
            }
            slice_sources.append(source)

        slice_dict = {
            "sources": slice_sources,
            "merging_method": {
                self.merging_method: self.merging_method_params
            }
        }
        
        slices.append(slice_dict)
        return slices
                    
    def merge_slices(self):
        self._pre_cache()
        self._build_tokenizer_and_embed()
        for cur_slice in self.slices:
            self._merge_slice(cur_slice)
        self._merge_postweights()
        self._update_output_config()
    
        if not self.in_memory:
            self._finalize_model()

    def fold_slices(self):
        logger.info("Start folding slices...")
        logger.info(f"Current slices: {self.slices}")
        self._pre_cache()
        self._build_tokenizer_and_embed()
        
        idx = len(self.slices) - 1
        has_merged = False # flag to check if 
        while idx >= 0:
            cur_slice = self.slices[idx]
            prev_slice = self.slices[idx - 1] if idx > 0 else None

            # Merge current slice if not done yet
            if not has_merged:
                self.current_layer_offset = idx
                self._merge_slice(cur_slice)
                self.current_layer_offset = idx
            has_merged = False

            # Collapse if layer is to be removed
            if "collapsing_method" in cur_slice.keys():
                logger.info(f"Collapsing layer {idx} to layer {idx - 1}")
                collapse_method, collapse_params = list(cur_slice["collapsing_method"].items())[0]
                order = cur_slice["merge_collapse_order"]

                # If layer i-1 is merged first (order == 0)
                if order == 0:
                    self.current_layer_offset = idx - 1
                    self._merge_slice(prev_slice)
                    self.current_layer_offset = idx - 1
                    has_merged = True
                else:
                    has_merged = False

                for prev_weight_name in self._get_matches_weight_names(str(idx - 1), match_layer=True):
                    if has_merged:
                        base_tensor = self._out_tensors[prev_weight_name]
                    else:
                        base_tensor = self.base_model_cache.get_tensor(prev_weight_name) if self.base_model else None
                    cur_weight_name = prev_weight_name.replace(f"layers.{idx - 1}.", f"layers.{idx}.")
                    donor_tensor = self._out_tensors[cur_weight_name]

                    # Collapse the tensor
                    self._merge_tensor(
                        prev_weight_name,
                        base_tensor,
                        [base_tensor, donor_tensor],
                        collapse_method,
                        collapse_params,
                    )

                    self._out_tensors.pop(cur_weight_name, None)
                
            idx -= 1

        self._merge_postweights()

        logger.info("_out_tensors after folding:")
        logger.info(self._out_tensors.keys())
        # Need to fix the tensor names and remove redundant tensors after collapsing
        index_map = {} # map to track the new indices of layers
        new_idx = 0
        # e.g. if we have 3 layers and the second layer is collapsed, we will have a mapping like {0: 0, 2: 1}
        for old_idx, slice in enumerate(self.slices):
            if "collapsing_method" in slice.keys():
                # This slice is removed -> no new index
                continue
            index_map[old_idx] = new_idx
            new_idx += 1
        # Update the _output_tensors with the retained layer indices
        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        new_tensors = {}
        for name, tensor in self._out_tensors.items():
            m = layer_pat.search(name)
            if m:
                old_idx = int(m.group(1))
                if old_idx in index_map.keys():
                    new_idx = index_map[old_idx]
                    new_name = name.replace(f"layers.{old_idx}.", f"layers.{new_idx}.")
                    new_tensors[new_name] = tensor
            else:
                new_tensors[name] = tensor     # embeddings, lm_head, etc.
        self._out_tensors = new_tensors
        self._output_config.num_hidden_layers = len(set(index_map.values()))
        logger.info("_out_tensors after layer index updating:")
        logger.info(self._out_tensors.keys())

        # Log the final number of layers, which layers are collapsed
        logger.info(f"Final number of layers: {self._output_config.num_hidden_layers}")
        logger.info(f"Layers retained: {set(index_map.keys())}")

        self._update_output_config()
    
        if not self.in_memory:
            self._finalize_model()
            
    def merge_models(self):
        # old version
        merging_models = [model["model"] for model in self.merging_models]
        default_merging_method_params = {
            param_name: values[0]['value'] 
            for param_name, values in self.merging_method_params.items()
        }    
        merged_res = self.merge_instance.merge(
            base_model=self.base_model,
            models_to_merge=merging_models,
            method_params=default_merging_method_params
        )
        self._aligned_tokenizer = merged_res["base_tokenizer"]
        self._save_checkpoint(merged_res, merging_models)
        
    def merge_models_v1(self):
        # new version 
        logger.info("merging models by slices")
        self.slices = self._convert_models_to_slices(self.merging_models)
        self._pre_cache()
        self._build_tokenizer_and_embed()
        for cur_slice in self.slices:
            self._merge_slice(cur_slice)
        self._merge_postweights()
        self._update_output_config()
        if not self.in_memory:
            self._finalize_model()

    
if __name__ == "__main__":
    pass
