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


DATASET_NAME = SupportedDatasets.VERB_AGREEMENT_TEST

dataloader = SFCDatasetLoader(DATASET_NAME, model, num_samples=10000,
                              local_dataset=True, base_folder_path=datapath)

experiment_name = 'sva_rc_test'
saving_dir = datapath / experiment_name

print(f'Using {SupportedDatasets.VERB_AGREEMENT_TEST} dataset and saving to the dir data/{experiment_name}.')


clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)


# Corrupted dataset here refers to the collection of patched prompts and their answers (verb completions) in the SFC paper terminology.

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

print('Corrupted dataset:')
for prompt in corrupted_dataset['prompt'][:3]:
  print("\nPrompt:", model.to_string(prompt), end='\n\n')
  
  for i, tok in enumerate(prompt):
    str_token = model.to_string(tok)
    print(f"({i-CONTROL_SEQ_LEN}, {str_token})", end=' ')
  print()


# Sanity checks

# Control sequence length must be the same for all samples in both datasets
clean_ds_control_len = clean_dataset['control_sequence_length']
corrupted_ds_control_len = corrupted_dataset['control_sequence_length']

assert torch.all(corrupted_ds_control_len == corrupted_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
assert torch.all(clean_ds_control_len == clean_ds_control_len[0]), "Control sequence length is not the same for all samples in the dataset"
assert clean_ds_control_len[0] == corrupted_ds_control_len[0], "Control sequence length is not the same for clean and corrupted samples in the dataset"
assert clean_dataset['answer'].max().item() < model.cfg.d_vocab, "Clean answers exceed vocab size"
assert corrupted_dataset['answer'].max().item() < model.cfg.d_vocab, "Patched answers exceed vocab size"
assert (clean_dataset['answer_pos'] < N_CONTEXT).all().item(), "Answer positions exceed logits length"
assert (corrupted_dataset['answer_pos'] < N_CONTEXT).all().item(), "Answer positions exceed logits length"


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

EXPERIMENT = 'sva_rc'

clear_cache()
sfc_model = SFC_Gemma(model, params_count=PARAMS_COUNT, control_seq_len=CONTROL_SEQ_LEN, 
                      attach_saes=RUN_WITH_SAES, caching_device=caching_device,
                      data_dir=datapath, experiment_name=EXPERIMENT)
clear_cache()

# sfc_model.print_saes()
# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Main part

# Here we'll call use CircuitEvaluator class, which encapsulates the SFC circuit evaluation logic.

from classes.sfc_evaluator import CircuitEvaluator

circuit_evaluator = CircuitEvaluator(sfc_model)


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

layers_to_extract = [
    # [18, 19, 20, 21, 22, 23, 24, 25],
    # [18, 19, 20, 21, 22, 23, 24],
    # [18, 19, 20, 21, 22, 23],
    # [18, 19, 20, 21, 22],
    [18, 19, 20, 21],
    [18, 19, 20],
    [18, 19],
    [18]
]
top_k_counts = [3, 5, 10, 25, 50, 100]
positions_to_select = [2, 6]  # Only consider features maximizing at these positions

extraction_params = {
    'abs_scores': False,
    'aggregation_type': 'max',
    'positions_to_select': positions_to_select,
    'include_components': [
                           'hook_resid_post', 
                           'attn.hook_z', 
                           'hook_mlp_out',
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
        
        # Display Neuronpedia visualizations for top features by score
        # display_neuronpedia_features(
        #     selected_nodes_info, 
        #     max_features_per_type=3,
        #     sfc_node_scores=circuit_evaluator.sfc_node_scores,
        #     abs_scores=abs_scores
        # )


# ## Running the ablation & restoration loop (variable top-k)

import pandas as pd
from collections import defaultdict

# ablate all resid error nodes up until (and including) this threshold 
ERRORS_LAYER_THRESHOLD_TOP = 17
ERRORS_LAYER_THRESHOLD_BOTTOM = 14

# Define error ablation function for this threshold
def ablate_error_hook(act_name):
    if 'hook_resid_post.hook_sae_error' not in act_name:
        return False
        
    # Split the input string by periods
    parts = act_name.split('.')
    error_layer_num = int(parts[1])
    
    return ERRORS_LAYER_THRESHOLD_BOTTOM <= error_layer_num <= ERRORS_LAYER_THRESHOLD_TOP

evaluation_params = {
    'cutoff_early_layers': False,
    'nodes_to_always_ablate': ablate_error_hook,
    'always_ablate_positions': positions_to_select,
    'nodes_to_always_ablate': ablate_error_hook,
    
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


for layer_idx, layer in enumerate(layers_to_extract):
    print(f"\n{'='*80}\nLAYER {layer}\n{'='*80}")
    
    for top_k_value in top_k_counts:
        print(f"\n{'-'*60}\nTop {top_k_value} features\n{'-'*60}")
        
        # Create a unique key for this configuration
        config_key = f"layer_{layer}_top_{top_k_value}"
        
        # Extract just this layer with the specified top_k value
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
            nodes_to_restore=nodes_to_restore,
            feature_indices_to_restore=feature_indices_dict,
            **evaluation_params
        )
        
        # Extract useful statistics from faithfulness_metrics
        mean_faithfulness = faithfulness_metrics.mean().item()
        std_faithfulness = faithfulness_metrics.std().item()
        
        # Store all the results
        results['layer'].append(layer)
        results['top_k'].append(top_k_value)
        results['num_selected_features'].append(sum(len(indices) for indices in feature_indices_dict.values()))
        results['circuit_size'].append(n_nodes_in_circuit)
        results['mean_faithfulness'].append(mean_faithfulness)
        results['std_faithfulness'].append(std_faithfulness)

        # Save the actual faithfulness metrics tensor for potential future analysis
        # detailed_results[config_key]['faithfulness_metrics'] = faithfulness_metrics.cpu().numpy()
        
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


# Create DataFrame from the collected results
results_df = pd.DataFrame(results)

SAVING_NAME_SUFFIX = 'layers_14_17_ablated'
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

def plot_faithfulness_vs_layer_ranges(results_df, layers_to_extract):
    """
    Plot faithfulness vs layer ranges, using the last layer in each range for x-axis positioning
    
    Args:
        results_df: DataFrame containing the results
        layers_to_extract: List of layer ranges used in the experiment
                           e.g., [[18, 19, 20, 21, 22], [18, 19, 20, 21], ...]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 7))
    
    # Extract the last (highest) layer from each range
    highest_layers = [max(layer_range) for layer_range in layers_to_extract]
    
    # Create x-axis labels that show the layer ranges
    x_labels = [f"{min(layer_range)}-{max(layer_range)}" if len(layer_range) > 1 
                else f"{layer_range[0]}" for layer_range in layers_to_extract]
    
    # Get unique top_k values
    top_k_values = sorted(results_df['top_k'].unique())
    
    # Set up colors for different top_k values
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_k_values)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # Different markers for different top_k values
    
    # For each top_k value, plot a line showing faithfulness vs layer range
    for i, k in enumerate(top_k_values):
        # Filter data for this top_k value
        k_data = results_df[results_df['top_k'] == k]
        
        # Create mapping from highest layer to row index
        layer_to_index = {highest_layers[i]: i for i in range(len(highest_layers))}
        
        # Prepare data points for plotting
        x_positions = []
        y_values = []
        y_errors = []
        
        for highest_layer in highest_layers:
            # Find the corresponding row in results_df
            matching_rows = k_data[k_data['layer'] == highest_layer]
            
            if not matching_rows.empty:
                # Get the first matching row
                row = matching_rows.iloc[0]
                
                x_positions.append(highest_layer)
                y_values.append(row['mean_faithfulness'])
                if 'std_faithfulness' in row:
                    y_errors.append(row['std_faithfulness'])
                else:
                    y_errors.append(0)
        
        if x_positions:
            # Plot the line
            plt.errorbar(x_positions, y_values, yerr=y_errors, 
                        marker=markers[i % len(markers)], color=colors[i],
                        linestyle='-', linewidth=2, markersize=8,
                        label=f'Top {k} features', capsize=5)
    
    # Customize the plot
    plt.xlabel('Layer Range', fontsize=14)
    plt.ylabel('Faithfulness', fontsize=14)
    plt.title('Faithfulness vs Layer Ranges for Different Feature Counts', fontsize=16)
    
    # Set custom x-ticks with range labels
    plt.xticks(highest_layers, x_labels, rotation=45)
    
    # Add grid, legend, etc.
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('faithfulness_vs_layer_ranges.png', dpi=300)
    plt.show()
    
    # Print summary statistics
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


plot_faithfulness_vs_layer_ranges(results_df, layers_to_extract)


# ## Running the ablation & restoration loop (variable upper-layer threshold)

import pandas as pd
from collections import defaultdict

error_layer_thresholds = [21, 22, 23, 24, 25]  # Different thresholds for error node ablation
top_k_value = 50  # Fixed value for top-k features

evaluation_params = {
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,
    
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


# for layer_idx, layer in enumerate(layers_to_extract):
#     print(f"\n{'='*80}\nLAYER {layer}\n{'='*80}")
    
#     # First, extract the features once (since we're using a fixed top_k)
#     selected_nodes_info = circuit_evaluator.sfc_node_scores.get_top_k_features_by_layer(
#         layer, 
#         top_k_counts=top_k_value,
#         verbose=False,  # Turn off built-in verbose output for cleaner display
#         **extraction_params
#     )
    
#     # Unpack the returned data
#     feature_indices_dict, maximizing_positions_dict = selected_nodes_info
    
#     # Loop over different error layer thresholds
#     for threshold in error_layer_thresholds:
#         print(f"\n{'-'*60}\nError Layer Threshold {threshold}\n{'-'*60}")
        
#         # Create a unique key for this configuration
#         config_key = f"layer_{layer}_top_{top_k_value}_error_threshold_{threshold}"
        
#         # Define error ablation function for this threshold
#         def ablate_error_hook(act_name):
#             if 'hook_resid_post.hook_sae_error' not in act_name:
#                 return False
                
#             # Split the input string by periods
#             parts = act_name.split('.')
#             error_layer_num = int(parts[1])
            
#             return error_layer_num <= threshold
        
#         # Update evaluation parameters with the current threshold function
#         current_eval_params = evaluation_params.copy()
#         current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
        
#         # Save the detailed feature information
#         detailed_results[config_key] = {
#             'feature_indices': feature_indices_dict,
#             'maximizing_positions': maximizing_positions_dict,
#             'layer': layer,
#             'top_k': top_k_value,
#             'error_threshold': threshold
#         }
        
#         # Prepare nodes_to_restore and feature_indices_to_restore for circuit evaluation
#         nodes_to_restore = list(feature_indices_dict.keys())  # All nodes with selected features
        
#         # Evaluate circuit faithfulness
#         faithfulness_metrics, n_nodes_in_circuit = circuit_evaluator.evaluate_circuit_faithfulness(
#             clean_dataset=clean_dataset,
#             patched_dataset=corrupted_dataset,
#             nodes_to_restore=nodes_to_restore,
#             feature_indices_to_restore=feature_indices_dict,
#             **current_eval_params
#         )
        
#         # Extract useful statistics from faithfulness_metrics
#         mean_faithfulness = faithfulness_metrics.mean().item()
#         std_faithfulness = faithfulness_metrics.std().item()
        
#         # Store all the results
#         results['layer'].append(layer)
#         results['top_k'].append(top_k_value)
#         results['error_threshold'].append(threshold)
#         results['num_selected_features'].append(sum(len(indices) for indices in feature_indices_dict.values()))
#         results['circuit_size'].append(n_nodes_in_circuit)
#         results['mean_faithfulness'].append(mean_faithfulness)
#         results['std_faithfulness'].append(std_faithfulness)
        
#         # Also collect position statistics
#         position_stats = {}
#         for node_name, positions in maximizing_positions_dict.items():
#             # Convert to numpy array for easier analysis
#             pos_array = positions.cpu().numpy()
#             position_stats[node_name] = {
#                 'position_counts': {int(pos): int(count) for pos, count in 
#                                    zip(*np.unique(pos_array, return_counts=True))}
#             }
        
#         detailed_results[config_key]['position_stats'] = position_stats
        
#         print(f"Layer {layer}, Error Threshold {threshold}: Mean faithfulness = {mean_faithfulness:.4f}")


# ### Plotting

# # Create DataFrame from the collected results
# results_df = pd.DataFrame(results)

# # Display the results
# print("\nSummary of results:")
# print(results_df)

# # If needed, convert layer values to proper types
# # This handles cases where layer might be a list or the last element of a range
# results_df['layer'] = results_df['layer'].apply(lambda x: x if isinstance(x, int) else max(x))

# # Save the results for future use
# results_df.to_csv(saving_dir / "faithfulness_with_error_thresholds.csv", index=False)

# # Save detailed results as pickle
# import pickle
# with open(saving_dir / "faithfulness_with_error_thresholds_detailed.pkl", "wb") as f:
#     pickle.dump(detailed_results, f)

# print(f"\nResults saved to 'faithfulness_with_error_thresholds.csv' and 'faithfulness_with_error_thresholds_detailed.pkl' within {saving_dir}")


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


# # Plot the heatmap
# heatmap_data, x_labels, thresholds = plot_faithfulness_heatmap(results_df, layers_to_extract)


# # Хуйотінг

import pandas as pd
from collections import defaultdict

# Define ranges for error layers to test - each loop will ablate ONE specific error layer
error_layers_to_ablate = list(range(26))  # From 0 to 25

evaluation_params = {
    'cutoff_early_layers': False,
    'always_ablate_positions': positions_to_select,
    
    'batch_size': 1024,
    'total_batches': None,
    'verbose': True
}

# Initialize data collection structures
results = defaultdict(list)


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


# Define window parameters
window_width = 4  # Initial window of 0-3
window_step = 1
max_right_bound = error_layers_to_ablate[-1]

# Initialize results
results = {
    'window_left': [],
    'window_right': [],
    'mean_faithfulness': [],
    'std_faithfulness': []
}

# Loop over sliding windows of error layers to ablate
window_left = error_layers_to_ablate[0]
window_right = window_width - 1

while window_right <= max_right_bound:
    print(f"\n{'-'*60}\nAblating Error Layers {window_left} to {window_right}\n{'-'*60}")
    
    # Define error ablation function for this window of error layers
    def ablate_error_hook(act_name):
        if 'hook_resid_post.hook_sae_error' not in act_name:
            return False
        
        # Split the input string by periods
        parts = act_name.split('.')
        error_layer_num = int(parts[1])
        
        # Ablate if the error layer is within the current window
        return window_left <= error_layer_num <= window_right
    
    # Update evaluation parameters with the current window ablation function
    current_eval_params = evaluation_params.copy()
    current_eval_params['nodes_to_always_ablate'] = ablate_error_hook
    
    # Evaluate circuit faithfulness
    faithfulness_metrics, n_nodes_in_circuit = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset=clean_dataset,
        patched_dataset=corrupted_dataset,
        **current_eval_params
    )
    
    # Extract useful statistics from faithfulness_metrics
    mean_faithfulness = faithfulness_metrics.mean().item()
    std_faithfulness = faithfulness_metrics.std().item()
    
    # Store all the results
    results['window_left'].append(window_left)
    results['window_right'].append(window_right)
    results['mean_faithfulness'].append(mean_faithfulness)
    results['std_faithfulness'].append(std_faithfulness)
    
    print(f"Layer {layer}, Error Layers {window_left}-{window_right}: Mean faithfulness = {mean_faithfulness:.4f}")
    
    # Slide the window by the step size
    window_left += window_step
    window_right += window_step


import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create a DataFrame for easier plotting with Plotly
df = pd.DataFrame({
    'Window Left': results['window_left'],
    'Window Right': results['window_right'],
    'Mean Faithfulness': results['mean_faithfulness'],
    'Std Faithfulness': results['std_faithfulness']
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
        gridcolor='lightgray'
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

# Print a summary table of the results
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Window':<15}{'Mean Faithfulness':<20}{'Std Faithfulness'}")
print("-" * 80)
for i in range(len(df)):
    print(f"{df['Window Label'].iloc[i]:<15}{df['Mean Faithfulness'].iloc[i]:<20.4f}{df['Std Faithfulness'].iloc[i]:.4f}")


# # Results analysis

# ## Looking at which token positions contain the most important features

# Sanity check: make sure I can reproduce the same top-k ranking as returned by the get_top_k_features_by_layer
LAYER = 22
act_to_check = f'blocks.{LAYER}.hook_resid_post.hook_sae_acts_post'

resid_latents = circuit_evaluator.sfc_node_scores.select_node_scores(lambda act: act == act_to_check)[act_to_check]


# POSITIONS = np.array(range(8))
POSITIONS = np.array([2, 6])
SELECT_K = 100
PRINT_K = 10

resid_latents_agg, maximizing_tok_positions = resid_latents[POSITIONS].max(dim=0)

top_values, top_features = resid_latents_agg.topk(SELECT_K)

for i in range(PRINT_K):
    print(f'Feature #{top_features[i]} with score = {top_values[i]:.6f} selected from position {POSITIONS[maximizing_tok_positions[top_features[i]]]}')


# This is consistent with the output we got from the `get_top_k_features_by_layer` method:

# ```
# hook_resid_post (100 features):
#   Indices: [10665, 1506, 4442, 15377, 1271] ... [299, 1841, 4651, 7909, 10213]
#   Maximizing positions distribution:
#     Position 2: 27 features (27.0%)
#     Position 6: 73 features (73.0%)
# 
#   Sample features and their maximizing positions:
#     Feature 10665: position 6, score 0.020020
#     Feature 1506: position 2, score 0.006042
#     Feature 4442: position 6, score 0.002594
#     Feature 15377: position 2, score 0.001610
#     Feature 1271: position 2, score 0.001534
# ```

import matplotlib.pyplot as plt
import numpy as np
import torch

# Move tensor to CPU and convert to numpy for plotting
positions = POSITIONS[maximizing_tok_positions[top_features].cpu().numpy()]

# Create the histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(positions, bins=np.arange(0.5, max(positions)+1.5, 1), 
                                 edgecolor='black', alpha=0.7)

# Add count labels on top of each bar
for i, count in enumerate(counts):
    if count > 0:
        plt.text(bins[i] + 0.5, count + 0.5, str(int(count)), 
                 ha='center', va='bottom', fontweight='bold')

# Customize the plot
plt.title(f'Histogram of Maximizing Token Positions for Resid SAE latents at layer {LAYER}', fontsize=14)
plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(range(1, int(max(positions))+1))
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print counts for each position
unique_positions, position_counts = np.unique(positions, return_counts=True)
for pos, count in zip(unique_positions, position_counts):
    percentage = 100 * count / len(positions)
    print(f"Position {int(pos)}: {count} features ({percentage:.1f}%)")


# The distribution of maximizing positions for our top K features is also the same as we got from `get_top_k_features_by_layer`.

# ## Setup x2 for running the notebook from this section (don't read)

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

def get_base_folder(parent_dir_name =  "tim-taras-sfc-errors"):
	# Find the project root dynamically
	current_dir = os.getcwd()
	while True:
		if os.path.basename(current_dir) == parent_dir_name:  # Adjust to match your project root folder name
			break
		parent = os.path.dirname(current_dir)
		if parent == current_dir:  # Stop if we reach the system root (failsafe)
			raise RuntimeError(f"Project root {parent_dir_name} not found. Check your folder structure.")
		current_dir = parent

	return current_dir

def get_data_path(base_folder=None, data_folder_name='data'):
	if base_folder is None:
		base_folder = get_base_folder()

	return Path(base_folder) / data_folder_name

def get_project_folder(base_folder=None, project_folder_name='sfc-errors'):
	if base_folder is None:
		base_folder = get_base_folder()
	
	return Path(base_folder) / project_folder_name

base_path = get_base_folder()
print(f"Base path: {base_path}")

project_path = get_project_folder(base_folder=base_path)
print(f"Project path: {project_path}")

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(base_path)
sys.path.append(str(project_path))

datapath = get_data_path(base_path) 
datapath


# ## Plots

# Load the CSV files with our metrics
EXPERIMENT = 'sva_rc_test'

standard_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_with_different_topk.csv")
modified_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_with_different_topk_layers_14_17_ablated.csv")


# ### Comparing restoration of only resid SAE nodes vs all SAE nodes

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

def plot_compare_faithfulness_vs_layer_ranges(results, results_modified, modified_title='modified', baseline=0):
    """
    Plot faithfulness vs layer ranges for both original and modified results using Plotly.
    
    Args:
        results: DataFrame containing the original results with integer 'layer' column
        results_modified: DataFrame containing the modified results with the same structure
        modified_title: String to describe the modified results in legends and labels (default: 'modified')
        baseline: Fixed value to subtract from all modified faithfulness scores (default: 0)
    """
    # Get unique layer values and top_k values
    all_layers = pd.concat([results['layer'], results_modified['layer']]).unique()
    top_k_values = sorted(results['top_k'].unique())
    
    # Sort layers
    all_layers = sorted(all_layers)
    
    # Create figure
    fig = go.Figure()
    
    # Set default Plotly colors
    colors = px.colors.qualitative.Plotly
    
    # For each top_k value, plot a line showing faithfulness vs layer range
    for i, k in enumerate(top_k_values):
        color = colors[i % len(colors)]  # Cycle through Plotly colors
        
        # Process original results
        k_data_orig = results[results['top_k'] == k]
        # Process modified results
        k_data_mod = results_modified[results_modified['top_k'] == k]
        
        # Prepare data points for plotting original results
        x_positions_orig = []
        y_values_orig = []
        y_errors_orig = []
        hover_texts_orig = []
        
        # Prepare data points for plotting modified results
        x_positions_mod = []
        y_values_mod = []
        y_errors_mod = []
        hover_texts_mod = []
        
        # Process each layer
        for layer in all_layers:
            # Find the corresponding row in original results
            matching_rows_orig = k_data_orig[k_data_orig['layer'] == layer]
            
            if not matching_rows_orig.empty:
                # Get the first matching row
                row = matching_rows_orig.iloc[0]
                
                x_positions_orig.append(layer)
                y_values_orig.append(row['mean_faithfulness'])
                y_errors_orig.append(row.get('std_faithfulness', 0))
                
                # Create hover text with additional information
                hover_text = f"Layer: {layer}<br>"
                hover_text += f"Top K: {k}<br>"
                hover_text += f"Mean Faithfulness: {row['mean_faithfulness']:.4f}<br>"
                hover_text += f"Std Deviation: {row.get('std_faithfulness', 0):.4f}<br>"
                hover_text += f"Selected Features: {row['num_selected_features']}"
                hover_texts_orig.append(hover_text)
            
            # Find the corresponding row in modified results
            matching_rows_mod = k_data_mod[k_data_mod['layer'] == layer]
            
            if not matching_rows_mod.empty:
                # Get the first matching row
                row = matching_rows_mod.iloc[0]
                
                # Subtract baseline from modified faithfulness score and normalize
                if np.abs(baseline) > 1e-3:
                    modified_faithfulness = (row['mean_faithfulness'] - baseline) / (1 - baseline)
                else:
                    modified_faithfulness = row['mean_faithfulness']
                
                x_positions_mod.append(layer)
                y_values_mod.append(modified_faithfulness)
                y_errors_mod.append(row.get('std_faithfulness', 0))
                
                # Create hover text with additional information
                hover_text = f"Layer: {layer}<br>"
                hover_text += f"Top K: {k}<br>"
                hover_text += f"Mean Faithfulness: {modified_faithfulness:.4f}<br>"
                hover_text += f"Original Value: {row['mean_faithfulness']:.4f}<br>"
                hover_text += f"Baseline: {baseline}<br>"
                if np.abs(baseline) > 1e-3:
                    hover_text += f"Normalization: (value - {baseline}) / (1 - {baseline})<br>"
                hover_text += f"Std Deviation: {row.get('std_faithfulness', 0):.4f}<br>"
                hover_text += f"Selected Features: {row['num_selected_features']}"
                hover_texts_mod.append(hover_text)
        
        # Plot original results
        if x_positions_orig:
            fig.add_trace(go.Scatter(
                x=x_positions_orig,
                y=y_values_orig,
                error_y=dict(
                    type='data',
                    array=y_errors_orig,
                    visible=True,
                    color=color,
                    thickness=1.5,
                    width=3
                ),
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(color=color, size=10, symbol='circle'),
                name=f'Original Top {k}',
                legendgroup=f'group{k}',
                hovertext=hover_texts_orig,
                hoverinfo='text'
            ))
        
        # Plot modified results with dashed lines but same color
        if x_positions_mod:
            fig.add_trace(go.Scatter(
                x=x_positions_mod,
                y=y_values_mod,
                error_y=dict(
                    type='data',
                    array=y_errors_mod,
                    visible=True,
                    color=color,
                    thickness=1.5,
                    width=3
                ),
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(color=color, size=10, symbol='square'),
                name=f'Top {k} when {modified_title}',
                legendgroup=f'group{k}',
                hovertext=hover_texts_mod,
                hoverinfo='text'
            ))
    
    # Customize the layout
    fig.update_layout(
        title={
            'text': 'Comparison of Faithfulness vs Layer Ranges for Different Feature Counts',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title='Layer',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Faithfulness',
            gridcolor='lightgray',
            range=[max(-0.05, min(min(results['mean_faithfulness']), min(results_modified['mean_faithfulness'] - baseline)) - 0.05), 
                   1.05]  # Dynamic range starting from slightly below min value
        ),
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600,
        # Move legend below the plot
        legend=dict(
            orientation='h',
            y=-0.15,  # Position below the plot
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=10)
        ),
        # Add more margin at the bottom to make room for the legend
        margin=dict(b=100)
    )
    
    # Add shapes to highlight differences better
    for i, k in enumerate(top_k_values):
        color = colors[i % len(colors)]
        k_data_orig = results[results['top_k'] == k]
        k_data_mod = results_modified[results_modified['top_k'] == k]
        
        for layer in all_layers:
            matching_orig = k_data_orig[k_data_orig['layer'] == layer]
            matching_mod = k_data_mod[k_data_mod['layer'] == layer]
            
            if not matching_orig.empty and not matching_mod.empty:
                y_orig = matching_orig['mean_faithfulness'].iloc[0]
                y_mod = matching_mod['mean_faithfulness'].iloc[0] - baseline
    
    # Add a note about baseline adjustment if applicable
    if np.abs(baseline) > 1e-3:
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Note: {modified_title} results normalized as (value - {baseline}) / (1 - {baseline})",
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center"
        )
    
    # Show the figure
    fig.show()
    
    # Print summary table
    print("Comparison of Faithfulness by Layer and Top K:")
    print("-" * 120)
    orig_label = "Original Mean"
    mod_label = f"{modified_title} Mean"
    if np.abs(baseline) > 1e-3:
        mod_label += f" (Normalized with baseline {baseline})"
    
    print(f"{'Layer':<10}{'Top K':<10}{orig_label:<20}{mod_label:<25}{'Difference':<15}{'% Change':<15}")
    print("-" * 120)
    
    for layer in all_layers:
        for k in top_k_values:
            orig_rows = results[(results['layer'] == layer) & (results['top_k'] == k)]
            mod_rows = results_modified[(results_modified['layer'] == layer) & (results_modified['top_k'] == k)]
            
            if not orig_rows.empty and not mod_rows.empty:
                orig_mean = orig_rows['mean_faithfulness'].iloc[0]
                mod_raw_mean = mod_rows['mean_faithfulness'].iloc[0]
                # Apply the same normalization as in the plot
                if np.abs(baseline) > 1e-3:
                    mod_adjusted_mean = (mod_raw_mean - baseline) / (1 - baseline)
                else:
                    mod_adjusted_mean = mod_raw_mean
                diff = mod_adjusted_mean - orig_mean
                pct_change = (diff / orig_mean) * 100 if orig_mean != 0 else float('inf')
                
                print(f"{layer:<10}{k:<10}{orig_mean:<20.4f}{mod_adjusted_mean:<25.4f}{diff:<15.4f}{pct_change:<15.2f}%")
    
    return fig


plot_compare_faithfulness_vs_layer_ranges(standard_results_df, modified_results_df, 
                                          modified_title='14-17 ablated', baseline=0.5662)


plot_compare_faithfulness_vs_layer_ranges(standard_results_df, modified_results_df, 
                                          modified_title='14-17 ablated')




