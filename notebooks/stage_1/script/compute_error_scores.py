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
GPU_TO_USE = 1

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


# ## Loading the model

# We'll work with Gemma-2 2B (base version)

from sae_lens import SAE, HookedSAETransformer, ActivationsStore

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


DATASET_NAME = SupportedDatasets.VERB_AGREEMENT

dataloader = SFCDatasetLoader(DATASET_NAME, model,
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


# # Setting up SFC

from classes.sfc_model import SFC_Gemma

RUN_WITH_SAES = False # we won't run the model with attached SAEs, computing the attr patching scores analytically

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
sfc_model.print_saes()

clear_cache()

# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Computing AtP scores

# Here we'll compute the SFC scores for all of the SAE nodes (although we'll be mainly interested in the error nodes)

batch_size = 100
total_batches = None

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()

# Below we'll call the main interface method for computing the SFC scores
clean_metric, patched_metric, node_scores = sfc_model.compute_sfc_scores_for_templatic_dataset(clean_dataset, corrupted_dataset, 
                                                                                              batch_size=batch_size, 
                                                                                              total_batches=total_batches,
                                                                                              run_without_saes=not RUN_WITH_SAES,
                                                                                              save_scores=True)

# Logit dif is the difference between the logit of the incorrect answers and patched answers
print(f'\nLogit dif on the clean tokens: {clean_metric}') # so this should be low and negative
print(f'\nLogit dif on the corrupted tokens: {patched_metric}') # and this should be high and positive


# # Results check

# #### Setup x2 for running the notebook from this section (don't read)

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


# Load all of our computed SFC scores and caches:

from classes.sfc_node_scores import SFC_NodeScores
EXPERIMENT = 'sva_rc'

sfc_scores = SFC_NodeScores(
    device=device,
    data_dir=datapath,
    experiment_name=EXPERIMENT,
    load_if_exists=True  # This will automatically load our computed scores
)
node_scores = sfc_scores.node_scores


# ### Studying the correlation with the layer depth

# Here I'll just check if our newly computed scores agree with my previous results (yes, they do)

ERROR_SCORE_THRESHOLD = 0.001
N_LAYERS = 26 

error_scores = []

for k in range(N_LAYERS):
    error_score = node_scores[f'blocks.{k}.hook_resid_post.hook_sae_error']
    error_scores.append(error_score)
    big_scores = torch.nonzero(error_score.abs() > ERROR_SCORE_THRESHOLD, as_tuple=True)[0]

    # Additionally, print the indices and values of the most weighty error scores
    if big_scores.numel() > 0:
        print(f'\n\nLayer #{k}:\n')
        values = error_score[big_scores]
        for i, v in zip(big_scores.tolist(), values.tolist()):
            print(f"Token position: {i}, Error node Value: {v}")


import plotly.express as px
import plotly.graph_objects as go
import torch
from scipy.stats import linregress

def plot_layer_correlation(tensor_list, token_to_plot=6, tensor_name='Tensor Values'):
    """
    Plots the correlation between tensor values at a specific token position and their corresponding layer indices.
    
    Args:
    - tensor_list (list of tensors): N_layers tensors of shape (seq_len,)
    - token_to_plot (int): The index of the token to extract values for plotting.
    - tensor_name (str): The name of the tensor for labeling the plot.
    """
    
    # Extract values at token_to_plot position
    values = [tensor[token_to_plot].item() for tensor in tensor_list]
    
    # Layer indices representing layer depth
    layer_indices = list(range(len(tensor_list)))

    # Compute regression line
    slope, intercept, r_value, _, _ = linregress(layer_indices, values)
    regression_line = [slope * idx + intercept for idx in layer_indices]

    # Create scatter plot with layer indices on x-axis and tensor values on y-axis
    fig = px.scatter(
        x=layer_indices,
        y=values,
        text=layer_indices,  # Label points with their layer index
        labels={'x': 'Layer Depth', 'y': tensor_name},
        title=f'Layer Depth vs {tensor_name} (r={r_value:.2f})'
    )

    # Add regression line
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=regression_line,
        mode='lines',
        name=f'Regression Line (r={r_value:.2f})',
        line=dict(color='red')
    ))

    # Show labels directly on the scatter points
    fig.update_traces(textposition='top center')

    # Show plot
    fig.show()


plot_layer_correlation(error_scores, tensor_name='Error Scores')


# Hmm that's interesting
# - error nodes don't seem to exhibit a clear linear trend w.r.t. layer depth
# - there's a strong peak around layer 17

import plotly.graph_objects as go

# Set the token position as the last token before the answer (as before)
token_to_plot = 6  

# Extract values at the specified token position
error_values = [tensor[token_to_plot].item() for tensor in error_scores]

# Layer indices representing layer numbers
layer_indices = list(range(len(error_scores)))

# Create the line plot
fig = go.Figure()

# Add Error Scores line
fig.add_trace(go.Scatter(
    x=layer_indices,
    y=error_values,
    mode='lines+markers',
    name='Error Scores'
))


# Update layout for better readability
fig.update_layout(
    title='Error Scores Across Layers',
    xaxis_title='Layer Number',
    yaxis_title='Values',
    legend_title='Metrics',
    template='plotly_white'
)

# Show the plot
fig.show()




