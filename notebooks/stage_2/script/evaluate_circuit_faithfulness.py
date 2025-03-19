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

dataloader = SFCDatasetLoader(DATASET_NAME, model, # num_samples=10000,
                              local_dataset=True, base_folder_path=datapath)


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

batch_size = 1024
total_batches = None
total_thresholds = 15

# Define threshold range (logarithmic scale) for SFC scores, which controls the number of nodes in the circuit
# (only the nodes above the threshold are kept in the circuit)
thresholds = np.concatenate([
    # A few samples below 0.0001
    np.logspace(-6, -4, int(total_thresholds * 0.2), endpoint=False),
    # Dense sampling in the more interesting region of [0.0001, 0.01]
    np.logspace(-4, -2, int(total_thresholds * 0.8)),
    # No samples above 0.01
])

# Print the thresholds to verify
print("Threshold values:")
for t in thresholds:
    print(f"{t:.6f}")


# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()


from tqdm.notebook import tqdm
import pandas as pd
from IPython.display import display

results = []
print("Evaluating standard circuit faithfulness...")

for i, threshold in enumerate(thresholds):
    print(f'Thresholds progress {i}/{len(thresholds)}')

    # Evaluate standard circuit (no special ablations)
    faithfulness, n_nodes = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset, 
        corrupted_dataset, 
        node_threshold=threshold,
        batch_size=batch_size,
        total_batches=total_batches
    )
    
    # Calculate average faithfulness and standard deviation
    avg_faithfulness = faithfulness.mean().item()
    std_faithfulness = faithfulness.std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': avg_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'standard'
    })

results_df = pd.DataFrame(results)
print(results_df)


save_dir = datapath / EXPERIMENT

results_df.to_csv(save_dir / "faithfulness_eval.csv", index=False)


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
        always_ablate_fn=always_ablate_fn,
        batch_size=batch_size,
        total_batches=total_batches
    )
    
    # Calculate average faithfulness and standard deviation
    avg_faithfulness = faithfulness.mean().item()
    std_faithfulness = faithfulness.std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': avg_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'resid_errors_ablated'
    })
    
# Convert results to DataFrame
resid_errors_ablated_results_df = pd.DataFrame(results)
print(resid_errors_ablated_results_df)


resid_errors_ablated_results_df.to_csv(save_dir / "faithfulness_eval_resid_err_abl.csv", index=False)


# ## Faithfulness eval when non-resid error nodes are ablated

always_ablate_fn = lambda name: 'hook_mlp_out.hook_sae_error' in name or 'hook_z.hook_sae_error' in name

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()


results = []
# Use the same thresholds as above
for i, threshold in enumerate(thresholds):
    print(f'Thresholds progress {i}/{len(thresholds)}')

    # Evaluate circuit with error nodes always ablated
    faithfulness, n_nodes = circuit_evaluator.evaluate_circuit_faithfulness(
        clean_dataset, 
        corrupted_dataset, 
        node_threshold=threshold,
        always_ablate_fn=always_ablate_fn,
        batch_size=batch_size,
        total_batches=total_batches
    )
    
    # Calculate average faithfulness and standard deviation
    avg_faithfulness = faithfulness.mean().item()
    std_faithfulness = faithfulness.std().item()
    
    results.append({
        'threshold': threshold,
        'n_nodes': n_nodes,
        'faithfulness': avg_faithfulness,
        'std_faithfulness': std_faithfulness,
        'circuit_type': 'errors_ablated'
    })
    
# Convert results to DataFrame
errors_ablated_results_df = pd.DataFrame(results)
print(errors_ablated_results_df)


errors_ablated_results_df.to_csv(save_dir / "faithfulness_eval_mlpattn_err_abl.csv", index=False)


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
EXPERIMENT = 'sva_rc_test'

standard_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval.csv")
resid_errors_ablated_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval_resid_err_abl.csv")
errors_ablated_results_df = pd.read_csv(datapath / EXPERIMENT / "faithfulness_eval_mlpattn_err_abl.csv")


# ## Plots

#  Now to the plotting part
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_faithfulness_single(results_df, circuit_type='standard'):
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
                array=results_df['std_faithfulness'],
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

def plot_multiple_faithfulness_results(dataframes_list, title="Circuit Faithfulness Comparison"):
    """
    Create a plot with multiple faithfulness lines from different dataframes.
    
    Args:
        dataframes_list: List of DataFrames, each containing faithfulness results
        title: Title for the plot
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Default Plotly colors
    plotly_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf'   # teal
    ]
    
    # Add a trace for each dataframe
    for i, df in enumerate(dataframes_list):
        # Extract circuit type for label
        circuit_type = df['circuit_type'].iloc[0] if 'circuit_type' in df.columns else f"Circuit {i+1}"
        
        # Sort by number of nodes for proper line plotting
        df_sorted = df.sort_values('n_nodes')
        
        # Get color for this circuit and create transparent version for error bands
        line_color = plotly_colors[i % len(plotly_colors)]
        # Convert hex to rgba with transparency
        rgba_color = f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.3)"
        
        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=df_sorted['n_nodes'],
                y=df_sorted['faithfulness'],
                mode='lines+markers',
                name=circuit_type,
                line=dict(color=line_color),
                legendgroup=f"group{i}",
                hovertemplate='Nodes: %{x}<br>Faithfulness: %{y:.3f}±%{error_y.array:.3f}'
            )
        )
        
        # Add error bars as a separate trace
        if 'std_faithfulness' in df_sorted.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['n_nodes'],
                    y=df_sorted['faithfulness'] + df_sorted['std_faithfulness'],
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
                    y=df_sorted['faithfulness'] - df_sorted['std_faithfulness'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=rgba_color,
                    fill='tonexty',
                    showlegend=False,
                    legendgroup=f"group{i}",
                    hoverinfo='skip'
                )
            )
    
    # Update layout
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


plot_multiple_faithfulness_results(
    [standard_results_df, resid_errors_ablated_results_df, errors_ablated_results_df], 
    title="Comparison of Circuit Faithfulness Across Configurations"
).show()


# Define a range of thresholds to test
test_thresholds = standard_results_df['threshold'].unique()

# Analyze circuit composition
composition_df = circuit_evaluator.analyze_circuit_composition(test_thresholds)
display(composition_df)




