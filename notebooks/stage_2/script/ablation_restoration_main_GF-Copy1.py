#!/usr/bin/env python
# coding: utf-8

# ## Setup

from IPython import get_ipython # type: ignore
ipython = get_ipython(); assert ipython is not None
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import einops
from jaxtyping import Float, Int
from torch import Tensor

torch.set_grad_enabled(False)

# Device setup
GPU_TO_USE = 3

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# utility to clear variables out of the memory & and clearing cuda cache
import gc
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


from pathlib import Path
import sys
import os

def get_base_folder(project_root = "tim-taras-sfc-errors"):
	# Find the project root dynamically
	current_dir = os.getcwd()
	while True:
		if os.path.basename(current_dir) == project_root:  # Adjust to match your project root folder name
			break
		parent = os.path.dirname(current_dir)
		if parent == current_dir:  # Stop if we reach the system root (failsafe)
			raise RuntimeError(f"Project root {project_root} not found. Check your folder structure.")
		current_dir = parent

	return current_dir

def get_project_folder(base_folder=None, project_folder_name='sfc-errors'):
	if base_folder is None:
		base_folder = get_base_folder()
	
	return Path(base_folder) / project_folder_name

def get_data_path(base_folder=None, data_folder_name='data'):
	if base_folder is None:
		base_folder = get_base_folder()

	return Path(base_folder) / data_folder_name


base_path = get_base_folder()
print(f"Base path: {base_path}")

project_path = get_project_folder(base_folder=base_path)
print(f"Project path: {project_path}")

sys.path.append(base_path)
sys.path.append(str(project_path))


datapath = get_data_path(base_path) 
datapath


# Additionally, setup the logging file so that we can track the output even when disconnected from the front-end Jupyter interface
import sys
import logging

nblog = open("nb.log", "a+")
sys.stdout.echo = nblog
sys.stderr.echo = nblog

get_ipython().log.handlers[0].stream = nblog
get_ipython().log.setLevel(logging.INFO)

get_ipython().run_line_magic('autosave', '5')


# ## Loading the model

# We'll work with Gemma-2 2B (base version)

from sae_lens import HookedSAETransformer

USE_INSTRUCT = False
PARAMS_COUNT = 2

MODEL_NAME = f'gemma-2-{PARAMS_COUNT}b' + ('-it' if USE_INSTRUCT else '')
print(f'Using {MODEL_NAME}')

model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
model


# ## Loading the data

# This uses my custom dataloader class, which parses raw data and prepares into a nice format for SFC, including providing some useful metadata such as token positions for where the answer should be, attention masks etc. The details of the class are convoluted because it was developed for a more general purpose than verb agreement tasks, so you can largely ignore the next few cells.

from classes.sfc_data_loader import SFCDatasetLoader
import utils.prompts as prompts
from utils.enums import *


# Load one of the supported dataset
DATASET_NAME = SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL

dataloader = SFCDatasetLoader(DATASET_NAME, model,
                              local_dataset=True, base_folder_path=datapath)
# Define the experiment directory where we'll save our data
EXPERIMENT = 'sva_rc_filtered'
saving_dir = datapath / EXPERIMENT

print(f'Using {DATASET_NAME} dataset and saving to the dir data/{EXPERIMENT}.')


# Below we'll parse the dataset into SFC-friendly format. `USE_SINGLE_DATASET` var below indicated whether to use a single dataset with no clean & patched dataset separation, or to use the original datasets from the SFC paper where we separate between clean & patched prompts for computing AtP/IG scores.

USE_SINGLE_DATASET = DATASET_NAME.value in [SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL.value, SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL_SALIENT_CIRCUIT.value]

if USE_SINGLE_DATASET:
    dataset = dataloader.get_single_dataset(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)
else:
    clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)


# Corrupted dataset here refers to the collection of patched prompts and their answers (verb completions) in the SFC paper terminology. It can be absent if we're a single dataset, which is why we write the cell below

if USE_SINGLE_DATASET:
    clean_dataset = dataset
    # we'll need to pass corrupted dataset as None to our main 
    # CircuitEvaluator.evaluate_circuit_faithfulness method below
    corrupted_dataset = None

    print(dataset)


CONTROL_SEQ_LEN = clean_dataset['control_sequence_length'][0].item() # how many first tokens to ignore when computing SFC scores
N_CONTEXT = clean_dataset['prompt'].shape[1]

CONTROL_SEQ_LEN, N_CONTEXT


print('Clean dataset:')
for prompt in clean_dataset['prompt'][:3]:
  print("\nPrompt:", model.to_string(prompt), end='\n\n')

  for i, tok in enumerate(prompt):
    str_token = model.to_string(tok)
    print(f"({i-CONTROL_SEQ_LEN}, {str_token})", end=' ')
  print()

if not USE_SINGLE_DATASET:
    print('Corrupted dataset:')
    for prompt in corrupted_dataset['prompt'][:3]:
      print("\nPrompt:", model.to_string(prompt), end='\n\n')
      
      for i, tok in enumerate(prompt):
        str_token = model.to_string(tok)
        print(f"({i-CONTROL_SEQ_LEN}, {str_token})", end=' ')
      print()


# Sanity checks
if not USE_SINGLE_DATASET:
    # Control sequence length must be the same for all samples in both datasets
    clean_ds_control_len = clean_dataset['control_sequence_length']
    corrupted_ds_control_len = corrupted_dataset['control_sequence_length']
    
    assert torch.all(corrupted_ds_control_len == corrupted_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
    assert torch.all(clean_ds_control_len == clean_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
    assert clean_ds_control_len[0] == corrupted_ds_control_len[0], "Control sequence length is not the same for clean and corrupted samples in the dataset"
else:
    assert dataset['true_answer'].max().item() < model.cfg.d_vocab, "Patched answers exceed vocab size"
    assert dataset['false_answer'].max().item() < model.cfg.d_vocab, "Patched answers exceed vocab size"
    assert (dataset['answer_pos'] < N_CONTEXT).all().item(), "Answer positions exceed logits length"


# # Setting up the SAEs

from classes.sfc_model import SFC_Gemma

RUN_WITH_SAES = True # we'll need to run the model with attached SAEs

# Determine the caching device, where we'll load our SAEs and compute the SFC scores
if RUN_WITH_SAES:
    caching_device = device 
else:
    caching_device = "cuda:0"


caching_device


# For replicating the SFC part from the paper I used my custom SFC_Gemma class. In short, it
# - Loads a Gemma model and its Gemma Scope SAEs (either attaching them to the model or not)
# - Provides interface methods to compute SFC scores (currently, only attr patching is supported) on an arbitrary dataset (that follows the format of my SFCDatasetLoader class from above)

clear_cache()
sfc_model = SFC_Gemma(model, params_count=PARAMS_COUNT, control_seq_len=CONTROL_SEQ_LEN, 
                      attach_saes=RUN_WITH_SAES, caching_device=caching_device,
                      data_dir=datapath, experiment_name=EXPERIMENT)
clear_cache()

# sfc_model.print_saes()
# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Main part

# Here we'll call use CircuitEvaluator class, which encapsulates the SFC circuit evaluation logic, as well as ablation & restoration functionality.

from classes.sfc_evaluator import CircuitEvaluator

circuit_evaluator = CircuitEvaluator(sfc_model)


# Define utilities for displaying the features we'll select to restore
def display_selected_features(selected_nodes_info, sfc_node_scores, 
                              abs_scores=False, print_k_positions=10):
    """
    Display the selected nodes and their features in a readable format
    
    Args:
        selected_nodes_info: Tuple containing (feature_indices_dict, maximizing_positions_dict)
        sfc_node_scores: SFC_NodeScores object containing the node scores
        abs_scores: Whether to use absolute values when displaying scores
    """
    feature_indices_dict, maximizing_positions_dict = selected_nodes_info
    
    print(f"Selected features from {len(feature_indices_dict)} activations in total")
    
    for node_name, indices in feature_indices_dict.items():
        # Get maximizing positions for this node
        max_positions = maximizing_positions_dict[node_name]
        
        # Display the node with its indices
        print(f"\n• {node_name} ({len(indices)} features):")
        
        # If there are many indices, display them in a compact way
        if len(indices) > 10:
            # Show first 5 and last 5 with ellipsis in between
            indices_display = f"{indices[:5]} ... {indices[-5:]}"
        else:
            indices_display = str(indices)
        
        print(f"  Indices: {indices_display}")
        
        # Get the actual scores for these indices to show their importance
        node_scores = sfc_node_scores.node_scores[node_name]
        
        if node_scores.dim() > 1:  # SAE feature node
            # Print histogram of maximizing positions
            position_counts = {}
            for pos in max_positions.cpu().numpy():
                position_counts[int(pos)] = position_counts.get(int(pos), 0) + 1
            
            print("  Maximizing positions distribution:")
            for pos, count in sorted(position_counts.items()):
                percentage = 100 * count / len(indices)
                print(f"    Position {pos}: {count} features ({percentage:.1f}%)")
            
            # Display sample feature-position mappings
            sample_size = min(print_k_positions, len(indices))
            if sample_size > 0:
                print("\n  Sample features and their maximizing positions:")
                for i in range(sample_size):
                    feat_idx = indices[i] if isinstance(indices, list) else indices.tolist()[i]
                    pos = max_positions[i].item()
                    
                    # Get the score at the maximizing position
                    if abs_scores:
                        score = abs(node_scores[pos, feat_idx].item())
                    else:
                        score = node_scores[pos, feat_idx].item()
                    
                    print(f"    Feature {feat_idx}: position {pos}, score {score:.6f}")

def display_neuronpedia_features(selected_nodes_info, model_name='gemma-2-2b', max_features_per_type=3, 
                                 sfc_node_scores=None, abs_scores=False):
    """
    Display selected SAE features using Neuronpedia visualizations
    
    Args:
        selected_nodes_info: Tuple containing (feature_indices_dict, maximizing_positions_dict)
        model_name: Name of the model in Neuronpedia
        max_features_per_type: Maximum number of features to display per type
        sfc_node_scores: SFC_NodeScores object for scoring features (required for score-based selection)
        abs_scores: Whether to use absolute scores
    """
    from IPython.display import IFrame, display
    
    feature_indices_dict, maximizing_positions_dict = selected_nodes_info
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    
    # Collect features by type with their scores
    resid_features = []
    mlp_features = []
    attn_features = []
    
    for node_name, indices in feature_indices_dict.items():
        # Extract layer number from node name
        layer_num = int(node_name.split('.')[1])
        
        # Get maximizing positions
        max_positions = maximizing_positions_dict[node_name]
        
        # Get feature scores if sfc_node_scores is provided
        if sfc_node_scores is not None:
            node_scores = sfc_node_scores.node_scores[node_name]
            
            # Extract scores for these indices at their maximizing positions
            feature_scores = []
            for i, idx in enumerate(indices if isinstance(indices, list) else indices.tolist()):
                max_pos = max_positions[i].item()
                score = node_scores[max_pos, idx].item()
                if abs_scores:
                    score = abs(score)
                feature_scores.append(score)
        else:
            # Default scores to placeholder values if not provided, assuming indices are already sorted by importance
            feature_scores = [len(indices) - i for i in range(len(indices))]
        
        # Determine component type
        if 'hook_resid_post' in node_name:
            component_type = 'resid'
            features = resid_features
            sae_width = '16k'
        elif 'hook_mlp_out' in node_name:
            component_type = 'mlp'
            features = mlp_features
            sae_width = '16k'
        elif 'attn.hook_z' in node_name:
            component_type = 'attn'
            features = attn_features
            sae_width = '16k'
        else:
            continue  # Skip unknown component types
        
        # Add features with their maximizing positions and scores
        for i, idx in enumerate(indices if isinstance(indices, list) else indices.tolist()):
            max_pos = max_positions[i].item()
            score = feature_scores[i]
            features.append((layer_num, idx, max_pos, sae_width, score))
    
    # Display residual features (sorted by score)
    if resid_features:
        # Sort by score (highest first)
        resid_features.sort(key=lambda x: x[4], reverse=True)
        
        print(f"\n{'='*80}\nRESIDUAL STREAM FEATURES (Top {min(max_features_per_type, len(resid_features))} by score)\n{'='*80}")
        for i, (layer, idx, max_pos, sae_width, score) in enumerate(resid_features[:max_features_per_type]):
            print(f"\nFeature {i+1}/{min(max_features_per_type, len(resid_features))}")
            print(f"Layer: {layer}, Feature: {idx}, Maximizing Position: {max_pos}, Score: {score:.6f}")
            
            # Generate Neuronpedia URL
            sae_release = f'{layer}-gemmascope-res-{sae_width}'
            url = html_template.format(model_name, sae_release, idx)
            link_without_query = url.split('?')[0]
            
            print('Link:', link_without_query)
            display(IFrame(url, width=1200, height=400))
    
    # Display MLP features (sorted by score)
    if mlp_features:
        # Sort by score (highest first)
        mlp_features.sort(key=lambda x: x[4], reverse=True)
        
        print(f"\n{'='*80}\nMLP FEATURES (Top {min(max_features_per_type, len(mlp_features))} by score)\n{'='*80}")
        for i, (layer, idx, max_pos, sae_width, score) in enumerate(mlp_features[:max_features_per_type]):
            print(f"\nFeature {i+1}/{min(max_features_per_type, len(mlp_features))}")
            print(f"Layer: {layer}, Feature: {idx}, Maximizing Position: {max_pos}, Score: {score:.6f}")
            
            # Generate Neuronpedia URL
            sae_release = f'{layer}-gemmascope-mlp-{sae_width}'
            url = html_template.format(model_name, sae_release, idx)
            link_without_query = url.split('?')[0]
            
            print('Link:', link_without_query)
            display(IFrame(url, width=1200, height=400))
    
    # For attention features (no visualization available yet)
    if attn_features:
        # Sort by score (highest first)
        attn_features.sort(key=lambda x: x[4], reverse=True)
        
        print(f"\n{'='*80}\nATTENTION FEATURES (Top {min(max_features_per_type, len(attn_features))} by score)\n{'='*80}")
        for i, (layer, idx, max_pos, _, score) in enumerate(attn_features[:max_features_per_type]):
            print(f"Layer: {layer}, Feature: {idx}, Maximizing Position: {max_pos}, Score: {score:.6f}")
            print("No Neuronpedia visualization available for attention features")


# ## Defining the set of SAE latents to restore

# Here we'll inspect the features we'll be restoring in the code. Restoring means:
# - Patching in original activations from unaffected pass (w/o ablations), into the forward pass that is affected by ablations.
# - i.e. patching in feature activations as they would have been without any ablations

# utility to generate feature restoration ranges
def generate_jagged_layer_list(start: int, end: int) -> list[list[int]]:
    """
    Generate a jagged nested list from [start, ..., end], 
    removing one item from the end in each subsequent list.

    Example:
        generate_jagged_layer_list(18, 25) returns:
        [
            [18, 19, 20, 21, 22, 23, 24, 25],
            [18, 19, 20, 21, 22, 23, 24],
            ...
            [18]
        ]
    """
    return [list(range(start, end - i + 1)) for i in range(end - start + 1)]

# Define the set of layers where we'll restore the features from
layers_to_extract = generate_jagged_layer_list(16, 22)
print(layers_to_extract)

top_k_counts = [1, 3, 5, 10, 25, 50, 100]
positions_to_select = [6]  # Only restore features at these token positions - same positions where we'll ablate the error nodes

# Config to determine how we choose top-K features
extraction_params = {
    'abs_scores': False, # Whether to use absolute values of scores when selecting top-K features.
    'aggregation_type': 'max', # within the given positions, how do we aggregate the features AtP score
    'positions_to_select': positions_to_select, # positions we select Top-K features from
    'include_components': [ # what type of features we select
                           'hook_resid_post', 
                           # 'attn.hook_z', 
                           # 'hook_mlp_out',
                          ]
}


# Store aggregation settings for consistent display
abs_scores = extraction_params['abs_scores']

for layer_idx, layer in enumerate(layers_to_extract):
    print(f"\n{'='*80}\nLAYER {layer}\n{'='*80}")
    
    for top_k_value in top_k_counts:
        print(f"\n{'-'*60}\nTop {top_k_value} features\n{'-'*60}")
        
        # Extract just this layer with the specified top_k value
        selected_nodes_info = circuit_evaluator.sfc_node_scores.get_top_k_features_by_layer(
            layer, 
            top_k_counts=top_k_value,
            verbose=False,  # Turn off built-in verbose output for cleaner display
            **extraction_params
        )
        
        # Display the selected nodes with their maximizing positions
        display_selected_features(selected_nodes_info, circuit_evaluator.sfc_node_scores, abs_scores, print_k_positions=0)
        
        # Optionally display Neuronpedia visualizations for top features by score
        # display_neuronpedia_features(
        #     selected_nodes_info, 
        #     max_features_per_type=3,
        #     sfc_node_scores=circuit_evaluator.sfc_node_scores,
        #     abs_scores=abs_scores
        # )


# ## Main ablation & restoration experiment (variable top-k)

# First experiment of the Ablation & Restoration experiments section of the post: ablate all error nodes within the specified layers and `positions_to_select` (last position 6 by default), and restore SAE nodes from different layer ranges (and for different number of Top-K features to restore)

import pandas as pd
from collections import defaultdict

# ablate all resid error nodes up until (and including) this threshold 
ERRORS_LAYER_THRESHOLD_TOP = 23
ERRORS_LAYER_THRESHOLD_BOTTOM = 0

# Define error ablation function for this threshold
def ablate_error_hook(act_name):
    if 'hook_resid_post.hook_sae_error' not in act_name:
        return False
        
    # Split the input string by periods
    parts = act_name.split('.')
    error_layer_num = int(parts[1])
    
    return ERRORS_LAYER_THRESHOLD_BOTTOM <= error_layer_num <= ERRORS_LAYER_THRESHOLD_TOP

# Consult the CircuitEvaluator.evaluate_circuit_faithfulness() method documentation for the meaning of the params below
evaluation_params = {
    'cutoff_early_layers': False,
    'nodes_to_always_ablate': ablate_error_hook,
    'always_ablate_positions': positions_to_select,
    # 'use_zero_ablation': True, # Uncomment if you want to run a zero ablation variant
    
    'batch_size': 1024,
    'total_batches': None,
    'verbose': True
}

# Initialize data collection structures
results = defaultdict(list)
detailed_results = {}  # For storing comprehensive data that doesn't fit in a dataframe

# Store aggregation settings for consistent display
abs_scores = extraction_params['abs_scores']


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


# Main loop
for layer_idx, layer in enumerate(layers_to_extract):
    print(f"\n{'='*80}\nLAYER {layer}\n{'='*80}")
    
    for top_k_value in top_k_counts:
        print(f"\n{'-'*60}\nTop {top_k_value} features\n{'-'*60}")
        
        # Create a unique key for this configuration
        config_key = f"layer_{layer}_top_{top_k_value}"
        
        # Select top-K features to restore
        selected_nodes_info = circuit_evaluator.sfc_node_scores.get_top_k_features_by_layer(
            layer, 
            top_k_counts=top_k_value,
            verbose=False,  # Turn off built-in verbose output for cleaner display
            **extraction_params
        )
        
        # Unpack the returned data
        feature_indices_dict, maximizing_positions_dict = selected_nodes_info
        
        # Save the detailed feature information
        detailed_results[config_key] = {
            'feature_indices': feature_indices_dict,
            'maximizing_positions': maximizing_positions_dict,
            'layer': layer,
            'top_k': top_k_value
        }
        
        # Display the selected nodes with their maximizing positions (optional)
        # display_selected_features(selected_nodes_info, circuit_evaluator.sfc_node_scores, abs_scores, print_k_positions=0)
        
        # Prepare nodes_to_restore and feature_indices_to_restore for circuit evaluation
        nodes_to_restore = list(feature_indices_dict.keys())  # All nodes with selected features
        
        # Evaluate circuit faithfulness
        faithfulness_metrics, n_nodes_in_circuit = circuit_evaluator.evaluate_circuit_faithfulness(
            clean_dataset=clean_dataset,
            patched_dataset=corrupted_dataset,
            # comment out these lines if you want to get the baseline faithfulness (without restoration)
            nodes_to_restore=nodes_to_restore,
            feature_indices_to_restore=feature_indices_dict,
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            return_all_metrics=True,
            **evaluation_params
        )
        
        # Before computing the mean and std metrics, filter out infs if there are any
        faithfulness_finite_mask = torch.isfinite(faithfulness_metrics)
        infinite_values_count = (~faithfulness_finite_mask).sum()
        if infinite_values_count > 0:
            print(f'Filtered out {infinite_values_count} infinite metrics')
        
        mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item()
        std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item()
        
        # Store all the results
        results['layer'].append(layer)
        results['top_k'].append(top_k_value)
        results['num_selected_features'].append(sum(len(indices) for indices in feature_indices_dict.values()))
        results['circuit_size'].append(n_nodes_in_circuit)
        results['mean_faithfulness'].append(mean_faithfulness)
        results['std_faithfulness'].append(std_faithfulness)

        # Save the actual faithfulness metrics tensor for potential future analysis
        metric_save_path = saving_dir / f'faithfulness_metrics_layer_{max(layer)}_topk_{top_k_value}.pt'
        torch.save(faithfulness_metrics, metric_save_path)
        
        # Also collect position statistics (if restoring features from different positions)
        position_stats = {}
        for node_name, positions in maximizing_positions_dict.items():
            # Convert to numpy array for easier analysis
            pos_array = positions.cpu().numpy()
            position_stats[node_name] = {
                'position_counts': {int(pos): int(count) for pos, count in 
                                   zip(*np.unique(pos_array, return_counts=True))}
            }
        
        detailed_results[config_key]['position_stats'] = position_stats
        
        # Compute overall position distribution
        all_positions = []
        for positions in maximizing_positions_dict.values():
            all_positions.extend(positions.cpu().numpy())
            
        if all_positions:
            position_counts = {int(pos): int(count) for pos, count in 
                              zip(*np.unique(all_positions, return_counts=True))}
            results['position_distribution'].append(str(position_counts))
        else:
            results['position_distribution'].append("{}")
        
        print(f"Layer {layer}, Top {top_k_value}: Mean faithfulness = {mean_faithfulness:.4f}")

        del faithfulness_metrics
        clear_cache()


# Create DataFrame from the collected results
results_df = pd.DataFrame(results)

SAVING_NAME_SUFFIX = 'GF_last_token'
SAVING_NAME_SUFFIX = '_' + SAVING_NAME_SUFFIX if SAVING_NAME_SUFFIX else SAVING_NAME_SUFFIX

SAVING_NAME_RESULTS = f'faithfulness_with_different_topk{SAVING_NAME_SUFFIX}.csv'
SAVING_NAME_DETAILED = f'faithfulness_with_different_topk_detailed{SAVING_NAME_SUFFIX}.csv'
# Display the results
print("\nSummary of results:")
print(results_df)

# If needed, convert layer values to proper types
# This handles cases where layer might be a list or the last element of a range
results_df['layer'] = results_df['layer'].apply(lambda x: x if isinstance(x, int) else max(x))

# Save the results for future use
results_df.to_csv(saving_dir / SAVING_NAME_RESULTS, index=False)

# Save detailed results as pickle
import pickle
with open(saving_dir / SAVING_NAME_DETAILED, "wb") as f:
    pickle.dump(detailed_results, f)

print(f"\nResults saved to {SAVING_NAME_RESULTS} and {SAVING_NAME_DETAILED} within {saving_dir}")


# ### Plotting

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px

def plot_faithfulness_vs_layer_ranges(results_df, layers_to_extract, plot_stds=False):
    """
    Plot faithfulness vs layer ranges using Plotly, with std shown as separate scatter points.
    
    Args:
        results_df: DataFrame with results (must include 'layer', 'top_k', 'mean_faithfulness', 'std_faithfulness')
        layers_to_extract: List of layer ranges (e.g. [[18,19,20], [20,21,22]])
    """
    import plotly.io as pio

    fig = go.Figure()

    highest_layers = [max(layer_range) for layer_range in layers_to_extract]
    x_labels = [f"{min(lr)}-{max(lr)}" if len(lr) > 1 else f"{lr[0]}" for lr in layers_to_extract]
    top_k_values = sorted(results_df['top_k'].unique())

    color_scale = px.colors.sequential.Viridis
    colors = color_scale[:len(top_k_values)]
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'triangle-left', 'triangle-right']

    for i, k in enumerate(top_k_values):
        k_data = results_df[results_df['top_k'] == k]
        x_positions = []
        y_values = []
        std_values = []
        x_for_std = []
        y_for_std = []

        for highest_layer in highest_layers:
            row = k_data[k_data['layer'] == highest_layer]
            if not row.empty:
                mean = row['mean_faithfulness'].iloc[0]
                std = row.get('std_faithfulness', pd.Series([0])).iloc[0]

                x_positions.append(highest_layer)
                y_values.append(mean)
                x_for_std.append(highest_layer)
                y_for_std.append(std)

        # Plot mean faithfulness as line+marker
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_values,
            mode='lines+markers',
            name=f"Top {k} features",
            marker_symbol=markers[i % len(markers)],
            marker_size=10,
            line=dict(width=2),
            customdata=np.stack([y_values], axis=-1),
            hovertemplate=(
                "Layer: %{x}<br>"
                "Mean Faithfulness: %{y:.3f}<extra>%{fullData.name}</extra>"
            )
        ))

        # Plot std values as separate scatter
        if plot_stds:
            fig.add_trace(go.Scatter(
                x=x_for_std,
                y=y_for_std,
                mode='markers',
                name=f"Top {k} features (std)",
                marker=dict(
                    size=8,
                    symbol='circle',
                    color=colors[i],
                    opacity=0.5
                ),
                showlegend=True,
                hovertemplate=(
                    "Layer: %{x}<br>"
                    "Std Faithfulness: %{y:.3f}<extra>%{fullData.name}</extra>"
                )
            ))

    # Set x-axis with custom tick labels
    fig.update_layout(
        title=f"Faithfulness vs Layer Ranges<br>Features are restored within the given layer range | Residual errors are ablated between layers {ERRORS_LAYER_THRESHOLD_BOTTOM} and {ERRORS_LAYER_THRESHOLD_TOP}",
        xaxis=dict(
            title="Layer Range",
            tickmode='array',
            tickvals=highest_layers,
            ticktext=x_labels
        ),
        yaxis_title="Faithfulness",
        legend_title="Top-K Features to restore",
        template="plotly_white",
        width=1000,
        height=600
    )

    # Save to PNG
    pio.write_image(fig, "faithfulness_vs_layer_ranges.png", format='png', scale=3, width=1000, height=600)

    fig.show()

    # Print textual summary
    print("Layer Range Faithfulness Summary:")
    for i, layer_range in enumerate(layers_to_extract):
        range_label = x_labels[i]
        highest_layer = highest_layers[i]

        print(f"\nLayer Range {range_label}:")
        for k in top_k_values:
            matching_rows = results_df[(results_df['layer'] == highest_layer) & (results_df['top_k'] == k)]
            if not matching_rows.empty:
                mean_faith = matching_rows['mean_faithfulness'].iloc[0]
                print(f"  Top {k} features: {mean_faith:.4f}")


results_df = pd.read_csv(saving_dir / SAVING_NAME_RESULTS)

plot_faithfulness_vs_layer_ranges(results_df, layers_to_extract, plot_stds=False)


# ## Ablation heatmap experiment (variable Ablation Error threshold)

# This is a follow-up experiment where we repeated a version of the previous experiment for different values of the ablation error threshold.

import pandas as pd
from collections import defaultdict

layers_to_extract = generate_jagged_layer_list(16, 25)
print(layers_to_extract)

# Different thresholds for error node ablation
error_layer_thresholds = list(range(16, 25 + 1))
top_k_value = 25  # Fixed value for top-k features

evaluation_params = {
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,
    # 'use_zero_ablation': True, # Uncomment if you want to run a zero ablation variant
    
    'batch_size': 1024,
    'total_batches': None,
    'verbose': False
}

# Initialize data collection structures
results = defaultdict(list)
detailed_results = {}  # For storing comprehensive data that doesn't fit in a dataframe

# Store aggregation settings for consistent display
abs_scores = extraction_params['abs_scores']


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()

circuit_evaluator = CircuitEvaluator(sfc_model)


for layer_idx, layer in enumerate(layers_to_extract):
    print(f"\n{'='*80}\nLAYER {layer}\n{'='*80}")
    
    # First, extract the features once (since we're using a fixed top_k)
    selected_nodes_info = circuit_evaluator.sfc_node_scores.get_top_k_features_by_layer(
        layer, 
        top_k_counts=top_k_value,
        verbose=False,  # Turn off built-in verbose output for cleaner display
        **extraction_params
    )
    
    # Unpack the returned data
    feature_indices_dict, maximizing_positions_dict = selected_nodes_info
    
    # Loop over different error layer thresholds
    for threshold in error_layer_thresholds:
        print(f"\n{'-'*60}\nError Layer Threshold {threshold}\n{'-'*60}")
        
        # Create a unique key for this configuration
        config_key = f"layer_{layer}_top_{top_k_value}_error_threshold_{threshold}"
        
        # Define error ablation function for this threshold
        def ablate_error_hook(act_name):
            if 'hook_resid_post.hook_sae_error' not in act_name:
                return False
                
            # Split the input string by periods
            parts = act_name.split('.')
            error_layer_num = int(parts[1])
            
            return error_layer_num <= threshold
        
        # Update evaluation parameters with the current threshold function
        current_eval_params = evaluation_params.copy()
        current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
        
        # Save the detailed feature information
        detailed_results[config_key] = {
            'feature_indices': feature_indices_dict,
            'maximizing_positions': maximizing_positions_dict,
            'layer': layer,
            'top_k': top_k_value,
            'error_threshold': threshold
        }
        
        # Prepare nodes_to_restore and feature_indices_to_restore for circuit evaluation
        nodes_to_restore = list(feature_indices_dict.keys())  # All nodes with selected features
        
        # Evaluate circuit faithfulness
        faithfulness_metrics, n_nodes_in_circuit = circuit_evaluator.evaluate_circuit_faithfulness(
            clean_dataset=clean_dataset,
            patched_dataset=corrupted_dataset,
            nodes_to_restore=nodes_to_restore,
            feature_indices_to_restore=feature_indices_dict,
            **current_eval_params
        )
        
        # Extract useful statistics from faithfulness_metrics
        mean_faithfulness = faithfulness_metrics.mean().item()
        std_faithfulness = faithfulness_metrics.std().item()
        
        # Store all the results
        results['layer'].append(layer)
        results['top_k'].append(top_k_value)
        results['error_threshold'].append(threshold)
        results['num_selected_features'].append(sum(len(indices) for indices in feature_indices_dict.values()))
        results['circuit_size'].append(n_nodes_in_circuit)
        results['mean_faithfulness'].append(mean_faithfulness)
        results['std_faithfulness'].append(std_faithfulness)
        
        # Also collect position statistics
        position_stats = {}
        for node_name, positions in maximizing_positions_dict.items():
            # Convert to numpy array for easier analysis
            pos_array = positions.cpu().numpy()
            position_stats[node_name] = {
                'position_counts': {int(pos): int(count) for pos, count in 
                                   zip(*np.unique(pos_array, return_counts=True))}
            }
        
        detailed_results[config_key]['position_stats'] = position_stats
        
        print(f"Layer {layer}, Error Threshold {threshold}: Mean faithfulness = {mean_faithfulness:.4f}")


# ### Plotting

# Create DataFrame from the collected results
results_df = pd.DataFrame(results)

SAVING_NAME_SUFFIX = 'GF'
SAVING_NAME_SUFFIX = '_' + SAVING_NAME_SUFFIX if SAVING_NAME_SUFFIX else SAVING_NAME_SUFFIX

SAVING_NAME_RESULTS = f'faithfulness_with_error_thresholds{SAVING_NAME_SUFFIX}.csv'
SAVING_NAME_DETAILED = f'faithfulness_with_error_thresholds_detailed{SAVING_NAME_SUFFIX}.pkl'

# Display the results
print("\nSummary of results:")
print(results_df)

# If needed, convert layer values to proper types
# This handles cases where layer might be a list or the last element of a range
results_df['layer'] = results_df['layer'].apply(lambda x: x if isinstance(x, int) else max(x))

# Save the results for future use
results_df.to_csv(saving_dir / SAVING_NAME_RESULTS, index=False)

# Save detailed results as pickle
import pickle
with open(saving_dir / SAVING_NAME_DETAILED, "wb") as f:
    pickle.dump(detailed_results, f)

print(f"\nResults saved to {SAVING_NAME_RESULTS} and {SAVING_NAME_DETAILED} within {saving_dir}")


def plot_faithfulness_heatmap(results_df, layer_ranges=None):
    """
    Create a heatmap showing faithfulness across layer ranges and error thresholds
    
    Args:
        results_df: DataFrame containing results with layer, error_threshold and mean_faithfulness columns
        layer_ranges: Optional list of layer ranges used in the experiment
                      e.g., [[18, 19, 20, 21, 22], [18, 19, 20, 21], ...]
                      If provided, x-axis labels will show ranges instead of just the max layer
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Extract unique layers and thresholds
    layers = sorted(results_df['layer'].unique())
    thresholds = sorted(results_df['error_threshold'].unique())
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(thresholds), len(layers)))
    
    # Fill the heatmap data
    for i, threshold in enumerate(thresholds):
        for j, layer in enumerate(layers):
            # Find the matching row in results_df
            matching_rows = results_df[(results_df['layer'] == layer) & 
                                      (results_df['error_threshold'] == threshold)]
            
            if not matching_rows.empty:
                heatmap_data[i, j] = matching_rows['mean_faithfulness'].iloc[0]
    
    # Create x-axis labels
    if layer_ranges is not None:
        # Create a mapping from max layer to range label
        layer_to_label = {}
        for layer_range in layer_ranges:
            max_layer = max(layer_range)
            if len(layer_range) > 1:
                label = f"{min(layer_range)}-{max(layer_range)}"
            else:
                label = str(layer_range[0])
            layer_to_label[max_layer] = label
        
        # Apply mapping to get labels in the correct order
        x_labels = [layer_to_label.get(layer, str(layer)) for layer in layers]
    else:
        x_labels = [str(layer) for layer in layers]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", 
                    xticklabels=x_labels, yticklabels=thresholds,
                    vmin=0, vmax=1.0, cbar_kws={'label': 'Faithfulness'})
    
    # Customize the plot
    plt.xlabel('Layer Range (Max Layer)', fontsize=12)
    plt.ylabel('Error Layer Threshold', fontsize=12)
    plt.title('Faithfulness Across Layer Ranges and Error Thresholds', fontsize=14)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('faithfulness_heatmap.png', dpi=300)
    plt.show()
    
    # Print summary of best configurations
    print("Top 5 configurations by faithfulness:")
    flat_data = []
    for i, threshold in enumerate(thresholds):
        for j, layer in enumerate(layers):
            flat_data.append({
                'layer': layers[j],
                'error_threshold': thresholds[i],
                'faithfulness': heatmap_data[i, j],
                'label': f"Layer {x_labels[j]}, Error Threshold {thresholds[i]}"
            })
    
    # Sort by faithfulness and display top 5
    flat_data.sort(key=lambda x: x['faithfulness'], reverse=True)
    for i, config in enumerate(flat_data[:5]):
        print(f"{i+1}. {config['label']}: {config['faithfulness']:.4f}")
    
    return heatmap_data, x_labels, thresholds


# Plot the heatmap
heatmap_data, x_labels, thresholds = plot_faithfulness_heatmap(results_df, layers_to_extract)


# ## (Errors) Sliding window ablation

# This is a section for reproducing our results from "Sliding window ablation" section

import pandas as pd
from collections import defaultdict

# Define ranges for error layers to test
# Use for defining ablation window config below
error_layers_to_ablate = list(range(26))  # From 0 to 25 # 26


evaluation_params_error = {
    # the params below should generally be fixed as they are
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,
    'return_all_metrics': True,

    # these params are modifable 
    # 'use_zero_ablation': True, # Uncomment if you want to run a zero ablation variant
    'batch_size': 1024,
    'total_batches': None,
    'verbose': True
}

# Initialize data collection structures
results = defaultdict(list)


# Initialize data collection structures
error_ablation_results = defaultdict(list)

# Define window parameters
window_width = 4
window_step = 1
max_right_bound = error_layers_to_ablate[-1]

# Loop over sliding windows of error layers to ablate
window_left = error_layers_to_ablate[0]
window_right = window_left + window_width - 1

window_left, window_right, window_right - window_left + 1


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


print("--- Running Sliding Window Ablation for Error Nodes ---")
while window_right <= max_right_bound:
    current_range = list(range(window_left, window_right + 1))
    print(f"Ablating Error Layers {current_range}...")
    
    # This function needs to be redefined in each loop iteration
    # to capture the current window_left and window_right values.
    def ablate_error_hook(act_name, left=window_left, right=window_right):
        if 'hook_resid_post.hook_sae_error' not in act_name:
            return False
        
        parts = act_name.split('.')
        error_layer_num = int(parts[1])
        return left <= error_layer_num <= right

    current_eval_params = evaluation_params_error.copy()
    current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
    
    faithfulness_metrics, _ = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset=clean_dataset,
        patched_dataset=corrupted_dataset,
        **current_eval_params
    )
    
    # Calculate Sum of SFC Scores for Error Nodes in Window
    window_sfc_score_sum = 0.0
    for node_name, scores_tensor in circuit_evaluator.sfc_node_scores.node_scores.items():
        if 'hook_resid_post.hook_sae_error' in node_name:
            layer_num = int(node_name.split('.')[1])
            if window_left <= layer_num <= window_right:
                window_sfc_score_sum += scores_tensor[positions_to_select].sum().item()
    
    faithfulness_finite_mask = torch.isfinite(faithfulness_metrics)
    mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item()
    std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item()

    # f_series = pd.Series(faithfulness_metrics[faithfulness_finite_mask])
    # print(f_series.describe())
    
    error_ablation_results['window_left'].append(window_left)
    error_ablation_results['window_right'].append(window_right)
    error_ablation_results['mean_faithfulness'].append(mean_faithfulness)
    error_ablation_results['std_faithfulness'].append(std_faithfulness)
    error_ablation_results['sfc_score_sum'].append(window_sfc_score_sum)
    
    window_left += window_step
    window_right += window_step
    clear_cache()

# Convert the results to a DataFrame
error_ablation_results_df = pd.DataFrame(error_ablation_results)
print("\n--- Error Node Ablation Summary ---")
print(error_ablation_results_df)


import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create a DataFrame for easier plotting with Plotly
df = pd.DataFrame({
    'Window Left': error_ablation_results_df['window_left'],
    'Window Right': error_ablation_results_df['window_right'],
    'Mean Faithfulness': error_ablation_results_df['mean_faithfulness'],
    'Std Faithfulness': error_ablation_results_df['std_faithfulness']
})

# Create window labels for the x-axis (e.g., "0-3", "1-4", etc.)
df['Window Label'] = df.apply(lambda row: f"{int(row['Window Left'])}-{int(row['Window Right'])}", axis=1)

# Calculate window midpoints for plotting
df['Window Midpoint'] = (df['Window Left'] + df['Window Right']) / 2

# Create the figure
fig = go.Figure()

# Add the line and markers
fig.add_trace(go.Scatter(
    x=df['Window Midpoint'],
    y=df['Mean Faithfulness'],
    mode='lines+markers',
    name='Mean Faithfulness',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=10, color='#1f77b4'),
))

# Add error bars
fig.add_trace(go.Scatter(
    x=np.concatenate([df['Window Midpoint'], df['Window Midpoint'][::-1]]),
    y=np.concatenate([
        df['Mean Faithfulness'] + df['Std Faithfulness'],
        (df['Mean Faithfulness'] - df['Std Faithfulness'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(color='rgba(255, 255, 255, 0)'),
    hoverinfo='skip',
    showlegend=False
))

# Customize the layout
fig.update_layout(
    title='Mean Faithfulness Across Sliding Windows of Error Layers',
    xaxis=dict(
        title='Error Layer Window',
        tickmode='array',
        tickvals=df['Window Midpoint'],
        ticktext=df['Window Label'],
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Mean Faithfulness',
        gridcolor='lightgray',
        dtick=0.2,
    ),
    hovermode='x unified',
    template='plotly_white',
    width=900,
    height=500,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Add custom hover text with more information
hover_text = [f"Window: {left}-{right}<br>Mean Faithfulness: {mean:.4f}<br>Std Deviation: {std:.4f}" 
              for left, right, mean, std in zip(
                  df['Window Left'], df['Window Right'], 
                  df['Mean Faithfulness'], df['Std Faithfulness']
              )]
fig.data[0].hovertext = hover_text
fig.data[0].hoverinfo = "text"

# Show the figure
fig.show()

pio.write_image(fig, "sliding_window_8_gf.png", format='png', scale=4, width=1000, height=500)

# Print a summary table of the results
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Window':<15}{'Mean Faithfulness':<20}{'Std Faithfulness'}")
print("-" * 80)
for i in range(len(df)):
    print(f"{df['Window Label'].iloc[i]:<15}{df['Mean Faithfulness'].iloc[i]:<20.4f}{df['Std Faithfulness'].iloc[i]:.4f}")


# ### Expanding window ablation

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


# #### Fixed Right Bound

import pandas as pd
from collections import defaultdict

# --- Configuration ---
FIXED_WINDOW_RIGHT = 25

# --- Data Collection ---
expanding_window_results = defaultdict(list)

print(f"\n{'='*80}\nRUNNING EXPANDING WINDOW ABLATION (ABLATING [L, {FIXED_WINDOW_RIGHT}])\n{'='*80}")

# --- New Loop Structure ---
# The left boundary 'L' will move from 0 up to the fixed right boundary.
for window_left in range(0, FIXED_WINDOW_RIGHT + 1):
    window_right = FIXED_WINDOW_RIGHT  # Right boundary is always fixed
    current_range = list(range(window_left, window_right + 1))
    print(f"Ablating Error Layers in range: {current_range}...")
    
    # Redefine the hook in each loop iteration to capture the current window values
    def ablate_error_hook(act_name, left=window_left, right=window_right):
        if 'hook_resid_post.hook_sae_error' not in act_name: return False
        parts = act_name.split('.'); error_layer_num = int(parts[1])
        return left <= error_layer_num <= right

    current_eval_params = evaluation_params_error.copy()
    current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
    
    # Evaluate faithfulness
    faithfulness_metrics, _ = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset=clean_dataset, patched_dataset=None, **current_eval_params
    )
    
    # Calculate Sum of SFC Scores for the ablated nodes
    window_sfc_score_sum = 0.0
    for node_name, scores_tensor in circuit_evaluator.sfc_node_scores.node_scores.items():
        if 'hook_resid_post.hook_sae_error' in node_name:
            layer_num = int(node_name.split('.')[1])
            if window_left <= layer_num <= window_right:
                window_sfc_score_sum += scores_tensor[positions_to_select].sum().item()
    
    # Process and store results
    faithfulness_finite_mask = torch.isfinite(faithfulness_metrics)
    mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item()
    std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item()
    
    expanding_window_results['window_left'].append(window_left)
    expanding_window_results['window_right'].append(window_right)
    expanding_window_results['mean_faithfulness'].append(mean_faithfulness)
    expanding_window_results['std_faithfulness'].append(std_faithfulness)
    expanding_window_results['sfc_score_sum'].append(window_sfc_score_sum)
    
    clear_cache()

# --- Display and Save Results ---
expanding_window_error_results_df = pd.DataFrame(expanding_window_results)
print("\n--- Expanding Window Error Ablation Summary ---")
print(expanding_window_error_results_df)

# Save the DataFrame
save_path = saving_dir / f"faithfulness_expanding_window_to_{FIXED_WINDOW_RIGHT}_GF.csv"
expanding_window_error_results_df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")


# ablate_errors_but_restore_features_GF_dataset.py

# --- Plotting Function for the Expanding Window Experiment ---

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_expanding_window_results(results_df):
    """
    Plots the results of the expanding window error ablation experiment.
    
    Args:
        results_df: DataFrame from the experiment, with 'window_left', 
                    'mean_faithfulness', and 'sfc_score_sum' columns.
    """
    if results_df.empty:
        print("No results to plot.")
        return
        
    # Create figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Faithfulness trace
    fig.add_trace(go.Scatter(
        x=results_df['window_left'],
        y=results_df['mean_faithfulness'],
        mode='lines+markers',
        name='Mean Faithfulness',
        line=dict(color='royalblue', width=2.5),
        marker=dict(size=8),
        hovertext=[f"Ablating [{l},{r}]<br>Faithfulness: {m:.3f} ± {s:.3f}" for l,r,m,s in zip(results_df.window_left, results_df.window_right, results_df.mean_faithfulness, results_df.std_faithfulness)],
        hoverinfo='text'
    ), secondary_y=False)

    # Add SFC Score Sum trace
    fig.add_trace(go.Scatter(
        x=results_df['window_left'],
        y=results_df['sfc_score_sum'],
        mode='lines+markers',
        name='Sum of Ablated SFC Scores',
        line=dict(color='firebrick', dash='dot', width=2),
        marker=dict(symbol='x-thin', size=8),
        hovertext=[f"Ablating [{l},{r}]<br>SFC Sum: {s:.2f}" for l,r,s in zip(results_df.window_left, results_df.window_right, results_df.sfc_score_sum)],
        hoverinfo='text'
    ), secondary_y=True)

    # Finalize Layout
    fixed_right_boundary = results_df['window_right'].iloc[0]
    fig.update_layout(
        title=f'<b>Faithfulness when Ablating Error Layers in Range [L, {fixed_right_boundary}]</b>',
        xaxis_title='Left Boundary of Ablation Window (L)',
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_xaxes(
        dtick=1,
        # Use 'window_left' for the range, matching the data being plotted
        range=[results_df['window_left'].min() - 0.5, results_df['window_left'].max() + 0.5] 
    )
    
    
    fig.update_yaxes(title_text="<b>Mean Faithfulness</b>", secondary_y=False, range=[-0.1, 1.05])
    fig.update_yaxes(title_text="<b>Sum of Ablated SFC Scores</b>", secondary_y=True)
    pio.write_image(fig, "expanding_left_window_gf.png", format='png', scale=4, width=1000, height=450)

    fig.show()

# --- Call the new plotting function ---
# Assumes `expanding_window_error_results_df` is available from the experiment cell.
plot_expanding_window_results(expanding_window_error_results_df)


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


# #### Fixed Left Bound

import pandas as pd
from collections import defaultdict

# --- Configuration ---
FIXED_WINDOW_LEFT = 0
MAX_RIGHT_BOUNDARY = sfc_model.model.cfg.n_layers - 1

# --- Data Collection ---
fixed_left_results = defaultdict(list)

print(f"\n{'='*80}\nRUNNING EXPANDING WINDOW ABLATION (ABLATING [{FIXED_WINDOW_LEFT}, R])\n{'='*80}")

# --- New Loop Structure ---
# The right boundary 'R' will move from the fixed left boundary up to the max layer.
for window_right in range(FIXED_WINDOW_LEFT, MAX_RIGHT_BOUNDARY + 1):
    window_left = FIXED_WINDOW_LEFT # Left boundary is always fixed
    current_range = list(range(window_left, window_right + 1))
    print(f"Ablating Error Layers in range: {current_range}...")
    
    def ablate_error_hook(act_name, left=window_left, right=window_right):
        if 'hook_resid_post.hook_sae_error' not in act_name: return False
        parts = act_name.split('.'); error_layer_num = int(parts[1])
        return left <= error_layer_num <= right

    current_eval_params = evaluation_params_error.copy()
    current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
    
    faithfulness_metrics, _ = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset=clean_dataset, patched_dataset=None, **current_eval_params
    )
    
    window_sfc_score_sum = 0.0
    for node_name, scores_tensor in circuit_evaluator.sfc_node_scores.node_scores.items():
        if 'hook_resid_post.hook_sae_error' in node_name:
            layer_num = int(node_name.split('.')[1])
            if window_left <= layer_num <= window_right:
                window_sfc_score_sum += scores_tensor[positions_to_select].sum().item()
    
    faithfulness_finite_mask = torch.isfinite(faithfulness_metrics)
    mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item()
    std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item()
    
    fixed_left_results['window_left'].append(window_left)
    fixed_left_results['window_right'].append(window_right)
    fixed_left_results['mean_faithfulness'].append(mean_faithfulness)
    fixed_left_results['std_faithfulness'].append(std_faithfulness)
    fixed_left_results['sfc_score_sum'].append(window_sfc_score_sum)
    
    clear_cache()

# --- Display and Save Results ---
fixed_left_error_results_df = pd.DataFrame(fixed_left_results)
print("\n--- Fixed-Left Expanding Window Error Ablation Summary ---")
print(fixed_left_error_results_df)

save_path = saving_dir / f"faithfulness_fixed_left_expanding_window_errors.csv"
fixed_left_error_results_df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")


# ablate_errors_but_restore_features_GF_dataset.py

# --- Plotting Function for the Fixed-Left Expanding Window Experiment ---

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_fixed_left_expanding_window_results(results_df):
    """
    Plots the results of the fixed-left expanding window error ablation experiment.
    
    Args:
        results_df: DataFrame from the experiment, with 'window_right', 
                    'mean_faithfulness', and 'sfc_score_sum' columns.
    """
    if results_df.empty:
        print("No results to plot.")
        return
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Faithfulness trace
    fig.add_trace(go.Scatter(
        x=results_df['window_right'],
        y=results_df['mean_faithfulness'],
        mode='lines+markers', name='Mean Faithfulness',
        line=dict(color='royalblue', width=2.5), marker=dict(size=8),
        hovertext=[f"Ablating [{l},{r}]<br>Faithfulness: {m:.3f} ± {s:.3f}" for l,r,m,s in zip(results_df.window_left, results_df.window_right, results_df.mean_faithfulness, results_df.std_faithfulness)],
        hoverinfo='text'
    ), secondary_y=False)

    # Add SFC Score Sum trace
    fig.add_trace(go.Scatter(
        x=results_df['window_right'],
        y=results_df['sfc_score_sum'],
        mode='lines+markers', name='Sum of Ablated SFC Scores',
        line=dict(color='firebrick', dash='dot', width=2), marker=dict(symbol='x-thin', size=8),
        hovertext=[f"Ablating [{l},{r}]<br>SFC Sum: {s:.2f}" for l,r,s in zip(results_df.window_left, results_df.window_right, results_df.sfc_score_sum)],
        hoverinfo='text'
    ), secondary_y=True)

    # Finalize Layout
    fixed_left_boundary = results_df['window_left'].iloc[0]
    fig.update_layout(
        title=f'<b>Faithfulness when Ablating Error Layers in Range [{fixed_left_boundary}, R]</b>',
        xaxis_title='Right Boundary of Ablation Window (R)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="<b>Mean Faithfulness</b>", secondary_y=False, range=[-0.1, 1.1])
    fig.update_yaxes(title_text="<b>Sum of Ablated SFC Scores</b>", secondary_y=True)

    fig.update_xaxes(
        title_text='Right Boundary of Ablation Window (R)',
        dtick=1,  # Force a tick mark for every integer.
        range=[results_df['window_right'].min() - 0.5, results_df['window_right'].max() + 0.5] # Clip axis to data range.
    )

    pio.write_image(fig, "expanding_right_window_gf.png", format='png', scale=4, width=1000, height=450)

    fig.show()

# --- Call the new plotting function ---
# Assumes `fixed_left_error_results_df` is available from the experiment cell.
plot_fixed_left_expanding_window_results(fixed_left_error_results_df) # 


# ## (Features) Sliding window ablation

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


# #### Top-K features ablation

# We didn't use this method in the post, but this section allows to ablate an arbitrary number of SAE features within the ablation window.

import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np

# Define window parameters
window_width = 4
WINDOW_STEP = 1
MAX_LAYER = error_layers_to_ablate[-1]

# Define the number of top features to ablate in each window.
TOP_K_TO_ABLATE = [] # 1, 5

# Parameters for selecting which resid_post features to ablate
FEATURE_EXTRACTION_PARAMS = {
    'abs_scores': False,
    'aggregation_type': 'max',
    'positions_to_select': positions_to_select, # Select features based on scores across all positions
    'include_components': ['hook_resid_post']
}

# Parameters for the faithfulness evaluation call
EVALUATION_PARAMS_FEATURES = {
    # the params below should generally be fixed as they are
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,
    'return_all_metrics': True,

    # these params are modifable 
    # 'use_zero_ablation': True, # Uncomment if you want to run a zero ablation variant
    'batch_size': 1024,
    'total_batches': None,
    'verbose': True
}

EVALUATION_PARAMS_FEATURES


# --- Main Sliding Window Feature Ablation Loop ---

# This dictionary will store a results DataFrame for each top_k value
feature_ablation_results_dict = {}

for top_k in TOP_K_TO_ABLATE:
    print(f"\n{'='*80}\nRUNNING FOR TOP_K = {top_k}\n{'='*80}")

    # Reset hooks before starting a new run for a different K
    sfc_model.model.reset_hooks()
    if RUN_WITH_SAES:
        sfc_model._reset_sae_hooks()
    clear_cache()
    
    # Initialize results for this specific K
    results = defaultdict(list)
    
    # Loop over sliding windows of layers
    window_left = error_layers_to_ablate[0]
    window_right = window_left + WINDOW_WIDTH - 1

    while window_right <= MAX_LAYER:
        current_layer_range = list(range(window_left, window_right + 1))
        print(f"\n---> Ablating Top {top_k} resid_post features from layers {current_layer_range}...")
        
        # 1. Select the top-K features from the current window's layers
        selected_nodes_info = circuit_evaluator.sfc_node_scores.get_top_k_features_by_layer(
            current_layer_range, 
            top_k_counts=top_k,
            verbose=False,
            **FEATURE_EXTRACTION_PARAMS
        )
        
        feature_indices_to_ablate, _ = selected_nodes_info
        num_ablated_features = sum(len(indices) for indices in feature_indices_to_ablate.values())
        print(f"      Selected {num_ablated_features} features to ablate.")
        
        # 2. Evaluate faithfulness by ablating these specific features
        faithfulness_metrics, _ = circuit_evaluator.evaluate_circuit_faithfulness(
            clean_dataset=clean_dataset,
            patched_dataset=corrupted_dataset,
            feature_indices_to_ablate=feature_indices_to_ablate,
            **EVALUATION_PARAMS_FEATURES
        )

        # Calculate Sum of SFC Scores for Selected Features
        total_sfc_score_sum = 0.0
        for node_name, indices in feature_indices_to_ablate.items():
            scores_tensor = circuit_evaluator.sfc_node_scores.node_scores[node_name]
            if len(indices) > 0:
                 # First, select the rows (positions). Then, from those, select the columns (features).
                 selected_scores = scores_tensor[positions_to_select, :][:, indices]
                 total_sfc_score_sum += selected_scores.sum().item()
        
        # 3. Process and store the results for this window
        faithfulness_finite_mask = torch.isfinite(faithfulness_metrics)
        mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item()
        std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item()
        
        results['window_left'].append(window_left)
        results['window_right'].append(window_right)
        results['mean_faithfulness'].append(mean_faithfulness)
        results['std_faithfulness'].append(std_faithfulness)
        results['sfc_score_sum'].append(total_sfc_score_sum) # Store the sum
        
        print(f"      Window {current_layer_range}: Mean faithfulness = {mean_faithfulness:.4f} ± {std_faithfulness:.4f}")
        
        # Slide the window for the next iteration
        window_left += WINDOW_STEP
        window_right += WINDOW_STEP
        
        del faithfulness_metrics, feature_indices_to_ablate
        clear_cache()
        
    # Store the completed DataFrame for this K value
    feature_ablation_results_dict[top_k] = pd.DataFrame(results)

# --- Save all results to CSV ---
for top_k, df in feature_ablation_results_dict.items():
    print(f"\n--- Sliding Window Feature Ablation Summary (Top {top_k}) ---")
    print(df)
    save_path = saving_dir / f"faithfulness_sliding_window_resid_features_top{top_k}.csv"
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


# ablate_errors_but_restore_features_GF_dataset.py

# --- Plotting function for the Fixed Top-K Sliding Window Experiment ---

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot_fixed_k_sliding_window(feature_results_dict):
    """
    Plots the results of the fixed top-k sliding window feature ablation.
    Shows faithfulness on the left Y-axis and the sum of SFC scores on the right Y-axis.
    
    Args:
        feature_results_dict: A dictionary where keys are top_k values and values are 
                              the corresponding result DataFrames.
    """
    if not feature_results_dict:
        print("No results to plot.")
        return
        
    # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    
    last_df = None # To set the x-axis ticks later
    for i, (top_k, df) in enumerate(feature_results_dict.items()):
        if df.empty:
            continue
        last_df = df
            
        color = colors[i % len(colors)]
        
        df['Window Label'] = df.apply(lambda row: f"{int(row['window_left'])}-{int(row['window_right'])}", axis=1)
        df['Window Midpoint'] = (df['window_left'] + df['window_right']) / 2
        
        # --- Faithfulness Trace (Left Y-Axis) ---
        fig.add_trace(go.Scatter(
            x=df['Window Midpoint'],
            y=df['mean_faithfulness'],
            mode='lines+markers',
            name=f'Ablate Top {top_k} Features',
            legendgroup=f'group_{top_k}',
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
            hovertext=[f"Faithfulness: {m:.3f} ± {s:.3f}" for m,s in zip(df.mean_faithfulness, df.std_faithfulness)],
            hoverinfo='text+name'
        ), secondary_y=False)
        
        # --- SFC Score Sum Trace (Right Y-Axis) ---
        fig.add_trace(go.Scatter(
            x=df['Window Midpoint'],
            y=df['sfc_score_sum'],
            mode='lines',
            name=f'SFC Score Sum (Top {top_k})',
            legendgroup=f'group_{top_k}',
            line=dict(color=color, dash='dot', width=2),
            hovertext=[f"SFC Score Sum: {s:.2f}" for s in df.sfc_score_sum],
            hoverinfo='text+name',
            showlegend=False # Hide this from the legend to reduce clutter
        ), secondary_y=True)

    # --- Finalize Layout ---
    title_text = 'Faithfulness & Ablated Score Sum for Fixed Top-K Feature Ablation'
    
    if last_df is not None:
        fig.update_layout(
            title=title_text,
            xaxis=dict(
                title='Feature Ablation Layer Window',
                tickmode='array',
                tickvals=last_df['Window Midpoint'],
                ticktext=last_df['Window Label'],
                gridcolor='lightgray',
                tickangle=45
            ),
            hovermode='x unified',
            template='plotly_white',
            legend_title_text='Ablation Setting'
        )
    else:
        fig.update_layout(title=title_text)
        
    # Set Y-axis titles
    fig.update_yaxes(title_text="<b>Mean Faithfulness</b>", secondary_y=False, range=[-0.1, 1.05])
    fig.update_yaxes(title_text="<b>Sum of Ablated SFC Scores</b>", secondary_y=True)
        
    fig.show()


# --- Call the rolled-back plotting function ---
if feature_ablation_results_dict:
    plot_fixed_k_sliding_window(feature_ablation_results_dict)


# #### Feature baselines implementation

# This is the variant we used in the post, implementing different feature baselines or `SELECTION_MODES` as named below.

# --- NEW Experiment: Feature Ablation via Layer-by-Layer Score Matching ---

import pandas as pd
from collections import defaultdict

# --- Configuration ---
# Define the three selection strategies we want to test
SELECTION_MODES = ['top_1', 'min_abs_diff', 'constrained_min_diff']

window_width = 4
WINDOW_STEP = 1
MAX_LAYER = error_layers_to_ablate[-1]

EVALUATION_PARAMS_FEATURES = {
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,

    'return_all_metrics': True,
    'batch_size': 1024,
    'total_batches': None,
    'verbose': True
}

# This dictionary will store the final DataFrame for each selection mode
all_feature_ablation_results = {}

# --- Main Loop over Selection Modes ---
for mode in SELECTION_MODES:
    print(f"\n{'='*80}\nRUNNING FEATURE ABLATION | SELECTION MODE: {mode}\n{'='*80}")
    
    sfc_model.model.reset_hooks()
    if RUN_WITH_SAES:
        sfc_model._reset_sae_hooks()
    clear_cache()

    mode_results = defaultdict(list)
    window_left, window_right = error_layers_to_ablate[0], error_layers_to_ablate[0] + WINDOW_WIDTH - 1

    while window_right <= MAX_LAYER:
        current_layer_range = list(range(window_left, window_right + 1))
        print(f"Processing window {current_layer_range}...")
        
        feature_indices_to_ablate = defaultdict(list)
        
        for layer_num in current_layer_range:
            # --- Common Setup for each layer ---
            error_node_name = f"blocks.{layer_num}.hook_resid_post.hook_sae_error"
            feature_node_name = f"blocks.{layer_num}.hook_resid_post.hook_sae_acts_post"
            
            if feature_node_name not in circuit_evaluator.sfc_node_scores.node_scores:
                print(f"  Warning: No scores for {feature_node_name}. Skipping layer.")
                continue
                
            feature_scores_at_pos = circuit_evaluator.sfc_node_scores.node_scores[feature_node_name][positions_to_select, :].sum(dim=0)
            
            # --- Mode-Specific Feature Selection Logic ---
            if mode == 'top_1':
                best_feature_idx = torch.argmax(feature_scores_at_pos).item()
                
            elif mode == 'min_abs_diff':
                target_score = circuit_evaluator.sfc_node_scores.node_scores[error_node_name][positions_to_select].sum().item()
                diffs = torch.abs(feature_scores_at_pos - target_score)
                best_feature_idx = torch.argmin(diffs).item()
                
            elif mode == 'constrained_min_diff':
                target_score = circuit_evaluator.sfc_node_scores.node_scores[error_node_name][positions_to_select].sum().item()
                candidate_indices = torch.where(feature_scores_at_pos >= target_score)[0]
                if len(candidate_indices) > 0:
                    candidate_scores = feature_scores_at_pos[candidate_indices]
                    best_feature_idx = candidate_indices[torch.argmin(candidate_scores)].item()
                else: # Fallback
                    best_feature_idx = torch.argmax(feature_scores_at_pos).item()
            
            feature_indices_to_ablate[feature_node_name].append(best_feature_idx)
            
        # ... (Evaluation and result storing logic is the same as before) ...
        actual_sfc_score_sum = 0.0;
        for node_name, indices in feature_indices_to_ablate.items():
            scores_tensor = circuit_evaluator.sfc_node_scores.node_scores[node_name];
            if len(indices) > 0:
                 selected_scores = scores_tensor[positions_to_select, :][:, indices];
                 actual_sfc_score_sum += selected_scores.sum().item()
                
        faithfulness_metrics, _ = circuit_evaluator.evaluate_circuit_faithfulness(clean_dataset=clean_dataset, patched_dataset=None, feature_indices_to_ablate=feature_indices_to_ablate, **EVALUATION_PARAMS_FEATURES);
        faithfulness_finite_mask = torch.isfinite(faithfulness_metrics); mean_faithfulness = faithfulness_metrics[faithfulness_finite_mask].mean().item(); std_faithfulness = faithfulness_metrics[faithfulness_finite_mask].std().item();
        mode_results['window_left'].append(window_left); mode_results['window_right'].append(window_right); mode_results['mean_faithfulness'].append(mean_faithfulness); mode_results['std_faithfulness'].append(std_faithfulness); mode_results['sfc_score_sum'].append(actual_sfc_score_sum);

        window_left += WINDOW_STEP; window_right += WINDOW_STEP
        clear_cache()
        
    # Store the final DataFrame for this mode
    all_feature_ablation_results[mode] = pd.DataFrame(mode_results)
    print(f"\n--- Feature Ablation Summary (Mode: {mode}) ---")
    print(all_feature_ablation_results[mode])


# ablate_errors_but_restore_features_GF_dataset.py

# --- NEW Plotting Function for Separate Faithfulness and Score Plots ---

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

def plot_multi_heuristic_ablation_separate(feature_results_dict, error_results_df):
    """
    Plots two separate graphs to compare sliding window ablation results:
    1. A graph comparing the Mean Faithfulness for each heuristic.
    2. A graph comparing the Sum of Ablated SFC Scores for each heuristic.
    
    Args:
        feature_results_dict: Dictionary where keys are heuristic names and values are result DataFrames.
        error_results_df: DataFrame with results from the error node ablation experiment.
    """
    # Create figures for each plot
    fig_faith = go.Figure()
    fig_scores = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    heuristic_names = {
        'top_1': 'Ablate Top-1 Feature',
        'min_abs_diff': 'Ablate Min Abs Diff Feature',
        'constrained_min_diff': 'Ablate Constrained Min Diff Feature'
    }
    
    last_df = None # To set the x-axis ticks later
    
    # --- 1. Add Traces for each Feature Selection Heuristic ---
    for i, (mode, df) in enumerate(feature_results_dict.items()):
        if df.empty: continue
        last_df = df
        color = colors[i % len(colors)]
        plot_name = heuristic_names.get(mode, mode)
        
        df['Window Label'] = df.apply(lambda r: f"{int(r['window_left'])}-{int(r['window_right'])}", axis=1)
        df['Window Midpoint'] = (df['window_left'] + df['window_right']) / 2
        
        # Add trace to Faithfulness plot
        fig_faith.add_trace(go.Scatter(
            x=df['Window Midpoint'], y=df['mean_faithfulness'], mode='lines+markers',
            name=plot_name, line=dict(color=color), marker=dict(size=8),
            hovertext=[f"Faithfulness: {m:.3f} ± {s:.3f}" for m,s in zip(df.mean_faithfulness, df.std_faithfulness)], hoverinfo='text+name'
        ))
        
        # Add trace to SFC Scores plot
        fig_scores.add_trace(go.Scatter(
            x=df['Window Midpoint'], y=df['sfc_score_sum'], mode='lines+markers',
            name=plot_name, line=dict(color=color), marker=dict(size=8),
            hovertext=[f"SFC Score Sum: {s:.2f}" for s in df.sfc_score_sum], hoverinfo='text+name'
        ))

    # --- 2. Add Traces for the Error Node Ablation Baseline ---
    if not error_results_df.empty:
        df_error = error_results_df
        df_error['Window Label'] = df_error.apply(lambda r: f"{int(r['window_left'])}-{int(r['window_right'])}", axis=1)
        df_error['Window Midpoint'] = (df_error['window_left'] + df_error['window_right']) / 2
        error_color = 'black'
        
        # Add to Faithfulness plot
        fig_faith.add_trace(go.Scatter(
            x=df_error['Window Midpoint'], y=df_error['mean_faithfulness'], mode='lines+markers',
            name='Ablate Error Nodes', line=dict(color=error_color, dash='dash'),
            marker=dict(symbol='x-thin', size=10, color=error_color),
            hovertext=[f"Faithfulness: {m:.3f} ± {s:.3f}" for m,s in zip(df_error.mean_faithfulness, df_error.std_faithfulness)], hoverinfo='text+name'
        ))

        # Add to SFC Scores plot
        fig_scores.add_trace(go.Scatter(
            x=df_error['Window Midpoint'], y=df_error['sfc_score_sum'], mode='lines+markers',
            name='Ablate Error Nodes', line=dict(color=error_color, dash='dash'),
            marker=dict(symbol='x-thin', size=10, color=error_color),
            hovertext=[f"SFC Score Sum: {s:.2f}" for s in df_error.sfc_score_sum], hoverinfo='text+name'
        ))
        
    # --- 3. Finalize Layouts for Both Plots ---
    xaxis_df = last_df if last_df is not None else (error_results_df if not error_results_df.empty else None)
    
    # Common layout settings
    common_layout = dict(hovermode='x unified', template='plotly_white', legend_title_text='Ablation Heuristic',
                        height=500)
    if xaxis_df is not None and not xaxis_df.empty:
        common_layout['xaxis'] = dict(
            title='Ablation Layer Window', tickmode='array',
            tickvals=xaxis_df['Window Midpoint'], ticktext=xaxis_df['Window Label'], tickangle=45
        )

    # Finalize Faithfulness Plot
    fig_faith.update_layout(title_text='<b>Faithfulness Comparison of Ablation Heuristics</b>', **common_layout)
    fig_faith.update_yaxes(title_text="Mean Faithfulness", range=[-0.1, 1.05])
    
    # Finalize SFC Scores Plot
    fig_scores.update_layout(title_text='<b>Sum of Ablated SFC Scores Comparison</b>', **common_layout)
    fig_scores.update_yaxes(title_text="Sum of SFC Scores")
    
    print("--- Displaying Faithfulness Comparison Plot ---")
    pio.write_image(fig_faith, "sliding_window_gf_full.png", format='png', scale=5, width=1000, height=500)
    fig_faith.show()
    
    print("\n--- Displaying Ablated SFC Score Sum Comparison Plot ---")
    pio.write_image(fig_scores, "sliding_window_gf_full_scores.png", format='png', scale=5, width=1000, height=500)
    fig_scores.show()


# --- Example Usage ---
# Assumes `all_feature_ablation_results` and `error_ablation_results_df` are available.
plot_multi_heuristic_ablation_separate(all_feature_ablation_results, error_ablation_results_df)


# ablate_errors_but_restore_features_GF_dataset.py

# --- Save All Experimental Results to CSV ---

# --- Configuration for Saving ---
# Modify this suffix to version your output files. It will be appended to filenames.
# For example, 'v2', 'with_zero_ablation', etc.
SAVING_NAME_SUFFIX = 'GF_last_token' 

# --- Prepare Suffix String ---
# Add a leading underscore to the suffix if it's not empty
suffix_str = f"_{SAVING_NAME_SUFFIX}" if SAVING_NAME_SUFFIX else ""

# --- 1. Save Feature Ablation Results ---
# This is a dictionary of DataFrames, so we loop through it.
print("\n--- Saving Feature Ablation Results ---")
if 'all_feature_ablation_results' in locals() and all_feature_ablation_results:
    for mode, df in all_feature_ablation_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Construct a descriptive filename including the mode
            filename = f"sliding_window_features_mode_{mode}{suffix_str}.csv"
            save_path = saving_dir / filename
            
            # Save the DataFrame to CSV
            df.to_csv(save_path, index=False)
            print(f"Saved results for mode '{mode}' to: {save_path}")
        else:
            print(f"No results to save for feature ablation mode '{mode}' (DataFrame is empty or not found).")
else:
    print("Variable 'all_feature_ablation_results' not found or is empty.")


# --- 2. Save Error Node Ablation Results ---
# This is a single DataFrame.
print("\n--- Saving Error Node Ablation Results ---")
if 'error_ablation_results_df' in locals() and isinstance(error_ablation_results_df, pd.DataFrame) and not error_ablation_results_df.empty:
    filename = f"sliding_window_errors{suffix_str}.csv"
    save_path = saving_dir / filename
    
    # Save the DataFrame to CSV
    error_ablation_results_df.to_csv(save_path, index=False)
    print(f"Saved error node ablation results to: {save_path}")
else:
    print("Variable 'error_ablation_results_df' not found or is empty. Nothing to save.")




