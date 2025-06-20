#!/usr/bin/env python
# coding: utf-8

# ## Setup

from IPython import get_ipython # type: ignore
ipython = get_ipython(); assert ipython is not None
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# Standard imports
import os

### Enforce determinism
# For CUDA >= 10.2
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)

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


DATASET_NAME = SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL

dataloader = SFCDatasetLoader(DATASET_NAME, model,
                              local_dataset=True, base_folder_path=datapath)


USE_SINGLE_DATASET = DATASET_NAME.value in [SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL.value, SupportedDatasets.VERB_AGREEMENT_TEST_CONFIDENT_MODEL_SALIENT_CIRCUIT.value]

if USE_SINGLE_DATASET:
    dataset = dataloader.get_single_dataset(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)
else:
    clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=False, prepend_generation_prefix=True)


if USE_SINGLE_DATASET:
    clean_dataset = dataset
    corrupted_dataset = None # for bwd compatibility

    print(dataset)


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
    caching_device = "cuda:3"


caching_device


# For replicating the SFC part from the paper I used my custom SFC_Gemma class. In short, it
# - Loads a Gemma model and its Gemma Scope SAEs (either attaching them to the model or not)
# - Provides interface methods to compute SFC scores (currently, only attr patching is supported) on an arbitrary dataset (that follows the format of my SFCDatasetLoader class from above)

EXPERIMENT = 'sva_rc_test'

clear_cache()
sfc_model = SFC_Gemma(model, params_count=PARAMS_COUNT, control_seq_len=CONTROL_SEQ_LEN, 
                      attach_saes=RUN_WITH_SAES, caching_device=caching_device,
                      data_dir=datapath, experiment_name=EXPERIMENT)
clear_cache()

# sfc_model.print_saes()
# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Evaluation part

# Here we'll call use CircuitEvaluator class, which encapsulates the SFC circuit evaluation logic.

from classes.sfc_evaluator import CircuitEvaluator
circuit_evaluator = CircuitEvaluator(sfc_model)


# ## Standard faithfulness eval

# We'll reproduce the original Figure 3 from the SFC paper, starting from the standard case of using the full circuit.

import numpy as np

# Define threshold range (logarithmic scale) for SFC scores, which controls the number of nodes in the circuit
total_thresholds = 20

a, b = 0.000027, 0.000032

thresholds = np.concatenate([
    # Coarse sampling below and above the zoomed-in region
    np.logspace(-5.5, np.log10(a), int(total_thresholds * 0.4), endpoint=False),
    
    # Dense sampling in the interesting region [a, b]
    np.logspace(np.log10(a), np.log10(b), int(total_thresholds * 0.2), endpoint=False),
    
    # Coarse sampling after the dense region
    np.logspace(np.log10(b), -3, int(total_thresholds * 0.4)),
])
# Or hard-code a specific threshold
# thresholds = [
#     thresholds[2] # 0.000139,
# ]

# Print the thresholds to verify
print("Threshold values:")
for t in thresholds:
    print(f"{t:.6f}")


nodes_per_threshold = []

for t in thresholds:
    # Initialize the ablation masks dictionary, mapping node names to their ablation masks
    ablation_masks = {}  
    n_nodes_in_circuit = 0
    
    threshold_ablation_masks = circuit_evaluator.determine_nodes_to_ablate(t)
            
    for node_name, mask in threshold_ablation_masks.items():
        ablation_masks[node_name] = mask

    # If cutoff_early_layers is True, we don't ablate the first layers of the model
    # early_layer_cutoff = self.model_wrapper.n_layers // 3  # First 1/3 of layers

    # for key in ablation_masks.keys():
    #     # Parse act name like this "blocks.5.hook_resid_post.hook_sae_acts_post"
    #     layer_str = key.split('.')[1]  # Gets "5" from the example
    #     layer_num = int(layer_str)
        
    #     # If node is from early layers, add to restore list and clear its ablation mask
    #     if layer_num < early_layer_cutoff:
    #         if key not in nodes_to_restore:
    #             nodes_to_restore.append(key)

    # Step 3: count how many nodes will be in the circuit (nodes that are not being ablated)
    for key, mask in ablation_masks.items():
        n_nodes_in_circuit += torch.sum(~mask).item()  # count the number of nodes for which mask is False (i.e. not ablated)

    nodes_per_threshold.append(n_nodes_in_circuit)

for n_nodes, t in zip(nodes_per_threshold, thresholds):
    print(n_nodes, ' - ', f'{t:.6f}') 


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


from tqdm.notebook import tqdm
import pandas as pd
from IPython.display import display

batch_size = 1024
total_batches = None

# --- This new list will store the detailed per-sample scores for later plotting ---
faithfulness_scores_by_threshold = []

# --- Modified Main Loop ---
results = []
N_OUTLIERS_TO_SHOW = 3 # How many top/bottom outliers to display
print("Evaluating standard circuit faithfulness and analyzing outliers...")

original_circuit_metrics_by_threshold = {}
for i, threshold in enumerate(thresholds):
    print(f"\n{'='*50}\nThreshold: {threshold:.6f} ({i+1}/{len(thresholds)})\n{'='*50}")

    faithfulness, n_nodes, circuit_m, full_m, empty_m = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset, 
        corrupted_dataset, 
        node_threshold=threshold, 
        
        batch_size=batch_size,
        total_batches=total_batches,
        
        verbose = True,
        return_all_metrics = True,
        _return_components_for_verification=True
    )
    
    # Store the circuit metrics for the current threshold
    original_circuit_metrics_by_threshold[threshold] = circuit_m

    if i == 0:
        # Construct the key that was just used to populate the cache
        # cache_key = (id(clean_dataset), id(corrupted_dataset), batch_size, total_batches)
        original_full_empty_metrics = empty_m # circuit_evaluator._empty_circuit_metrics_cache[cache_key]
        original_full_model_metrics = full_m
        print("Captured the original circuit metrics for verification. Shape ", original_full_empty_metrics.shape)
    
    # Store all scores for this threshold for later histogram plotting
    faithfulness_scores_by_threshold.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'scores': faithfulness.cpu()
    })
        
    # Filter out inf/nan for robust analysis
    finite_mask = torch.isfinite(faithfulness)
    finite_scores = faithfulness[finite_mask]
    finite_indices = torch.where(finite_mask)[0]
    
    print(f"Total scores: {len(faithfulness)}, Finite scores: {len(finite_scores)}")
    
    # Calculate mean and std on the filtered scores
    mean_faithfulness = finite_scores.mean().item()
    std_faithfulness = finite_scores.std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': mean_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'standard'
    })


import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple

def save_full_experiment_results(
    faithfulness_scores: List[Dict[str, Any]],
    circuit_metrics: Dict[float, torch.Tensor],
    full_model_metrics: torch.Tensor,
    empty_circuit_metrics: torch.Tensor,
    data_dir: str,
    experiment_name: str,
    filename: str = "full_experiment_results.pt"
):
    """
    Saves all key data structures from a faithfulness evaluation run into a single file.

    This includes:
    - The per-sample faithfulness scores for each threshold.
    - The per-sample circuit metrics (m(C)) for each threshold.
    - The per-sample full model metrics (m(M)).
    - The per-sample empty circuit metrics (m(∅)).

    Args:
        faithfulness_scores: The 'faithfulness_scores_by_threshold' list.
        circuit_metrics: The 'original_circuit_metrics_by_threshold' dictionary.
        full_model_metrics: The 'original_full_model_metrics' tensor.
        empty_circuit_metrics: The 'original_full_empty_metrics' tensor.
        data_dir: The base data directory path.
        experiment_name: The name of the specific experiment sub-folder.
        filename: The name of the file to save.
    """
    # Bundle all data into a single dictionary
    results_bundle = {
        'faithfulness_scores_by_threshold': faithfulness_scores,
        'circuit_metrics_by_threshold': circuit_metrics,
        'full_model_metrics': full_model_metrics,
        'empty_circuit_metrics': empty_circuit_metrics
    }
    
    # Construct the full path
    experiment_path = Path(data_dir) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    filepath = experiment_path / filename
    
    print(f"Saving full experiment results bundle to: {filepath}")
    torch.save(results_bundle, filepath)
    print("Save complete.")
    
    return faithfulness_scores, circuit_metrics, full_model_metrics, empty_circuit_metrics

save_full_experiment_results(
    faithfulness_scores=faithfulness_scores_by_threshold,
    circuit_metrics=original_circuit_metrics_by_threshold,
    full_model_metrics=original_full_model_metrics,
    empty_circuit_metrics=original_full_empty_metrics,
    data_dir=datapath,
    experiment_name=EXPERIMENT,
    filename='faith_filtered_full.pt'
)


results_df = pd.DataFrame(results)
results_df.to_csv(datapath / EXPERIMENT / "faithfulness_eval.csv")

print("\n--- Aggregated Results ---")
print(results_df)


# ## Faithfulness eval when resid error nodes are ablated

always_ablate_fn = lambda name: 'hook_resid_post.hook_sae_error' in name

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()


results = []
print("Evaluating standard circuit faithfulness with resid error nodes ablated...")

# Use the same thresholds as above
for i, threshold in enumerate(thresholds):
    print(f'Thresholds progress {i}/{len(thresholds)}')
    
    # Evaluate circuit with error nodes always ablated
    faithfulness, n_nodes = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset, 
        corrupted_dataset, 
        node_threshold=threshold,
        nodes_to_always_ablate=always_ablate_fn,
        batch_size=batch_size,
        total_batches=total_batches,
        verbose = i == 0, # log only for the first batch
        return_all_metrics = True
    )
    
    # Before computing the mean and std metrics, filter out infs which seem to be pathological
    faithfulness_finite_mask = torch.isfinite(faithfulness)
    infinite_values_count = (~faithfulness_finite_mask).sum()
    if infinite_values_count > 0:
        print(f'Filtered out {infinite_values_count} infinite metrics')
    
    mean_faithfulness = faithfulness[faithfulness_finite_mask].mean().item()
    std_faithfulness = faithfulness[faithfulness_finite_mask].std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': mean_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'resid_errors_ablated'
    })
    
# Convert results to DataFrame
resid_errors_ablated_results_df = pd.DataFrame(results)
print(resid_errors_ablated_results_df)


save_dir = datapath / EXPERIMENT


resid_errors_ablated_results_df.to_csv(save_dir / "faithfulness_eval_resid_err_abl.csv", index=False)


# ## Faithfulness eval when non-resid error nodes are ablated

always_ablate_fn = lambda name: 'hook_mlp_out.hook_sae_error' in name or 'hook_z.hook_sae_error' in name

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()


results = []
# Use the same thresholds as above
for i, threshold in enumerate(thresholds):
    print(f'Thresholds progress {i}/{len(thresholds)}')

    # Evaluate circuit with error nodes always ablated
    faithfulness, n_nodes = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset, 
        corrupted_dataset, 
        node_threshold=threshold,
        nodes_to_always_ablate=always_ablate_fn,
        batch_size=batch_size,
        total_batches=total_batches,
        verbose = i == 0, # log only for the first batch
        return_all_metrics = True
    )
    
    # Before computing the mean and std metrics, filter out infs which seem to be pathological
    faithfulness_finite_mask = torch.isfinite(faithfulness)
    infinite_values_count = (~faithfulness_finite_mask).sum()
    if infinite_values_count > 0:
        print(f'Filtered out {infinite_values_count} infinite metrics')
    
    mean_faithfulness = faithfulness[faithfulness_finite_mask].mean().item()
    std_faithfulness = faithfulness[faithfulness_finite_mask].std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': mean_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'errors_ablated'
    })
    
# Convert results to DataFrame
errors_ablated_results_df = pd.DataFrame(results)
print(errors_ablated_results_df)


save_dir = datapath / EXPERIMENT

errors_ablated_results_df.to_csv(save_dir / "faithfulness_eval_mlpattn_err_abl.csv", index=False)


# Define a range of thresholds to test
test_thresholds = results_df['threshold'].unique()

# Analyze circuit composition
composition_df = circuit_evaluator.analyze_circuit_composition(test_thresholds)
display(composition_df)


# # Results analysis

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


# Load the CSV files with our metrics
EXPERIMENT = 'sva_rc_filtered' # 'sva_rc_filtered'

standard_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval.csv")
resid_errors_ablated_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval_resid_err_abl.csv")
errors_ablated_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval_mlpattn_err_abl.csv")


N_SAMPLES = clean_dataset['prompt'].shape[0]
N_SAMPLES


# ## Plots

#  Now to the plotting part
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_faithfulness_single(results_df, circuit_type='standard', n_samples = N_SAMPLES):
    """
    Create plots similar to those in the SFC paper, with error bars.
    
    Args:
        results_df: DataFrame with evaluation results including std_faithfulness
        circuit_type: Which circuit type to plot (default: 'standard')
    
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Sort by number of nodes for proper line plotting
    results_df = results_df.sort_values('n_nodes')
    
    # Add faithfulness trace with error bars
    fig.add_trace(
        go.Scatter(
            x=results_df['n_nodes'],
            y=results_df['faithfulness'],
            mode='lines+markers',
            name='Faithfulness',
            error_y=dict(
                type='data',
                array=results_df['std_faithfulness'] / np.sqrt(n_samples),
                visible=True,
                thickness=1,
                width=3
            ),
            hovertemplate='Nodes: %{x}<br>Faithfulness: %{y:.3f}±%{error_y.array:.3f}'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Circuit Faithfulness vs. Number of Nodes" + f' ({circuit_type.title()})',
        xaxis_title="Number of Nodes in Circuit",
        yaxis_title="Faithfulness",
        template="plotly_white",
        width=900,
        height=600,
        hovermode="closest"
    )
    
    return fig


plot_faithfulness_single(standard_results_df, circuit_type='standard').show()


plot_faithfulness_single(resid_errors_ablated_results_df, circuit_type='resid errors ablated').show()


plot_faithfulness_single(errors_ablated_results_df, circuit_type='mlp/attn errors ablated').show()


import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.io as pio

def plot_multiple_faithfulness_results(dataframes_list, plot_stds=None, title="Circuit Faithfulness Comparison", n_samples=1):
    """
    Create a plot with multiple faithfulness lines from different dataframes.

    Args:
        dataframes_list: List of DataFrames, each containing faithfulness results
        plot_stds: List of booleans, one per dataframe, indicating whether to plot std shading
        title: Title for the plot
        n_samples: Used if needed for error scaling

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    if plot_stds is None:
        plot_stds = [True] * len(dataframes_list)

    for i, df in enumerate(dataframes_list):
        circuit_type = df['circuit_type'].iloc[0] if 'circuit_type' in df.columns else f"Circuit {i+1}"
        df_sorted = df.sort_values('n_nodes')

        line_color = plotly_colors[i % len(plotly_colors)]
        rgba_color = f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.3)"

        if 'std_faithfulness' in df_sorted.columns:
            custom_data = np.stack([df_sorted['std_faithfulness']], axis=-1)
            hover_template = (
                'Nodes: %{x}<br>'
                'Faithfulness: %{y:.3f}<br>'
                'Std: %{customdata[0]:.3f}<extra>%{fullData.name}</extra>'
            )
        else:
            custom_data = None
            hover_template = (
                'Nodes: %{x}<br>'
                'Faithfulness: %{y:.3f}<extra>%{fullData.name}</extra>'
            )

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=df_sorted['n_nodes'],
                y=df_sorted['faithfulness'],
                mode='lines+markers',
                name=circuit_type,
                line=dict(color=line_color),
                legendgroup=f"group{i}",
                customdata=custom_data,
                hovertemplate=hover_template
            )
        )

        # Conditionally add std shading
        if plot_stds[i] and 'std_faithfulness' in df_sorted.columns:
            upper = df_sorted['faithfulness'] + df_sorted['std_faithfulness']
            lower = df_sorted['faithfulness'] - df_sorted['std_faithfulness']

            fig.add_trace(
                go.Scatter(
                    x=df_sorted['n_nodes'],
                    y=upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=f"group{i}",
                    hoverinfo='skip'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_sorted['n_nodes'],
                    y=lower,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=rgba_color,
                    fill='tonexty',
                    showlegend=False,
                    legendgroup=f"group{i}",
                    hoverinfo='skip'
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Nodes in Circuit",
        yaxis_title="Faithfulness",
        legend_title="Circuit Type",
        template="plotly_white",
        width=900,
        height=600,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


standard_results_df['circuit_type'] = 'Standard'
errors_ablated_results_df['circuit_type'] = 'MLP & Attn errors ablated'
resid_errors_ablated_results_df['circuit_type'] = 'Resid errors ablated'


fig = plot_multiple_faithfulness_results(
    [standard_results_df, resid_errors_ablated_results_df, errors_ablated_results_df], 
    plot_stds=[False, False, False],
    title="Comparison of Circuit Faithfulness Across Configurations"
)
pio.write_image(fig, "faithfulness_original_comparison.png", format='png', scale=3, width=900, height=600)

fig.show()


plot_multiple_faithfulness_results(
    [standard_results_df], 
    title="Comparison of Circuit Faithfulness Across Configurations"
).show()




