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


# - Corrupted dataset here refers to the collection of patched prompts and their answers (verb completions) in the SFC paper terminology.
# - The datasets support padding but currently its logic doesn't work well with templatic datasets like this, gotta fix it later **TODO**

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

from classes.sfc_model import *

RUN_WITH_SAES = True # we'll run the model with attached SAEs to automatically compute error scores and patch w.r.t. them

# So there will be no caching device (everything we'll be done on a single GPU)
if RUN_WITH_SAES:
    caching_device = device 
else:
    caching_device = "cuda:2"


caching_device


# I'll use my custom `SFC_Gemma` class. In short, it
# - Loads a Gemma model and its Gemma Scope SAEs (either attaching them to the model or not)
# - Provides interface methods for computing SFC scores **and activation patching** scores

clear_cache()

sfc_model = SFC_Gemma(model, params_count=PARAMS_COUNT, control_seq_len=CONTROL_SEQ_LEN, 
                      attach_saes=RUN_WITH_SAES, caching_device=caching_device)
sfc_model.print_saes()

clear_cache()

# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Activation patching

# Define the patching config

N_LAYERS = sfc_model.n_layers

layers_to_patch={
        'resid': list(range(N_LAYERS)),
        'mlp': list(range(N_LAYERS)),
        'attn': list(range(N_LAYERS))
    }
token_specific_error_types = ['resid', 'mlp', 'attn']
token_positions = [2, 3, -2]


batch_size = 900
total_batches = None

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()
clear_cache()

# Compute the patching effects for each type of error (resid, mlp, attn)
patching_effects = sfc_model.compute_act_patching_scores_for_errors(clean_dataset, corrupted_dataset, 
                                                                    layers_to_patch=layers_to_patch, 
                                                                    token_specific_error_types=token_specific_error_types,
                                                                    token_positions=token_positions,
																	batch_size=batch_size, total_batches=total_batches)

patching_effects.keys()


# Each error type should have the number of "patching effects" equal to the number of layers
patching_effects['resid'].shape, patching_effects['mlp'].shape, patching_effects['attn'].shape 


def save_dict(data_dict, dataset_name='sva', prefix=''):   
    filename = f'{dataset_name}_{prefix}_act_patching_scores.pkl'
     
    print(f'Saving {filename}...')
    filename = datapath / filename

    # Save using torch.save, which properly handles tensor storage details
    torch.save(data_dict, filename)


save_dict(patching_effects['resid'], prefix='resid')
save_dict(patching_effects['mlp'], prefix='mlp') 
save_dict(patching_effects['attn'], prefix='attn') 


# # Patching results analysis

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

base_path = get_base_folder()
print(f"Base path: {base_path}")

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(base_path)

datapath = get_data_path(base_path) 
datapath


# Load error node scores:
# - SFC scores (attribution patching)
# - Activation patching scores

def load_act_patching_scores(dataset_name='sva', prefix=''):
    filename = f'{dataset_name}_{prefix}_act_patching_scores.pkl'
    print(f'Loading {filename}...')
    filename = datapath / filename

    # Use torch.load with map_location to force loading on the desired device
    data_dict = torch.load(filename, map_location=torch.device(device))
    return data_dict

def load_sfc_scores(experiment_name='sva_rc', device=device):
	from classes.sfc_node_scores import SFC_NodeScores

	sfc_scores = SFC_NodeScores(
		device=device,
		data_dir=datapath,
		experiment_name=experiment_name,
		load_if_exists=True  # This will automatically load our computed scores
	)
	return sfc_scores.node_scores

sfc_scores = load_sfc_scores()

resid_patching_scores = load_act_patching_scores(prefix='resid')
mlp_patching_scores = load_act_patching_scores(prefix='mlp') 
attn_patching_scores = load_act_patching_scores(prefix='attn') 

resid_patching_scores.shape, mlp_patching_scores.shape, attn_patching_scores.shape


N_LAYERS = 26 

# Extract the error scores from the scores dict of all SFC nodes
resid_sfc_scores = [sfc_scores[f'blocks.{k}.hook_resid_post.hook_sae_error'] for k in range(N_LAYERS)]
mlp_sfc_scores = [sfc_scores[f'blocks.{k}.hook_mlp_out.hook_sae_error'] for k in range(N_LAYERS)]
attn_sfc_scores = [sfc_scores[f'blocks.{k}.attn.hook_z.hook_sae_error'] for k in range(N_LAYERS)]


# ### Plotting all of the act-patching and SFC scores together

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_scores(
    resid_sfc_scores, 
    mlp_sfc_scores, 
    attn_sfc_scores,
    resid_patching_scores, 
    mlp_patching_scores, 
    attn_patching_scores,
    token_position=None,
    title_suffix="",
    subplot_titles=("SFC Scores", "Patching Scores")
):
    """
    Plot SFC scores and patching scores across layers.
    
    Parameters:
    -----------
    resid_sfc_scores, mlp_sfc_scores, attn_sfc_scores : list of tensors
        Lists of tensors containing SFC scores for residual, MLP, and attention components.
    resid_patching_scores, mlp_patching_scores, attn_patching_scores : list of tensors
        Lists of tensors containing patching scores for residual, MLP, and attention components.
    token_position : int or None
        If int, selects that specific token position from each tensor.
        If None, computes mean across all token positions.
        If -2, selects the second-to-last token (useful for handling padding).
    title_suffix : str
        Additional text to append to the plot title.
    subplot_titles : tuple of str
        Titles for the two subplots.
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Process the tensors based on token_position
    if token_position is not None:
        # Extract values at specific token position
        resid_sfc_values = [tensor[token_position].float().cpu().numpy() for tensor in resid_sfc_scores]
        mlp_sfc_values = [tensor[token_position].float().cpu().numpy() for tensor in mlp_sfc_scores]
        attn_sfc_values = [tensor[token_position].float().cpu().numpy() for tensor in attn_sfc_scores]
        
        resid_patching_values = [tensor[token_position].float().cpu().numpy() for tensor in resid_patching_scores]
        mlp_patching_values = [tensor[token_position].float().cpu().numpy() for tensor in mlp_patching_scores]
        attn_patching_values = [tensor[token_position].float().cpu().numpy() for tensor in attn_patching_scores]
        
        position_desc = f"at token position {token_position}"
    else:
        # Compute mean across all positions
        resid_sfc_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in resid_sfc_scores]
        mlp_sfc_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in mlp_sfc_scores]
        attn_sfc_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in attn_sfc_scores]
        
        resid_patching_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in resid_patching_scores]
        mlp_patching_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in mlp_patching_scores]
        attn_patching_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in attn_patching_scores]
        
        position_desc = "averaged across token positions"
    
    # Layer indices representing layer numbers
    layer_indices = list(range(len(resid_sfc_scores)))
    
    # Create subplots
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)
    
    # Define colors manually to ensure consistency
    colors = {'Residual': 'blue', 'MLP': 'red', 'Attention': 'green'}
    
    # Add Residual, MLP, and Attention Error Scores to first subplot
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=resid_sfc_values,
        mode='lines+markers',
        name='Residual Error Scores',
        line=dict(color=colors['Residual'])
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=mlp_sfc_values,
        mode='lines+markers',
        name='MLP Error Scores',
        line=dict(color=colors['MLP'])
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=attn_sfc_values,
        mode='lines+markers',
        name='Attention Output Scores',
        line=dict(color=colors['Attention'])
    ), row=1, col=1)
    
    # Add Residual, MLP, and Attention Patching Scores to second subplot with same colors
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=resid_patching_values,
        mode='lines+markers',
        name='Residual Patching Scores',
        line=dict(color=colors['Residual'])
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=mlp_patching_values,
        mode='lines+markers',
        name='MLP Patching Scores',
        line=dict(color=colors['MLP'])
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=layer_indices,
        y=attn_patching_values,
        mode='lines+markers',
        name='Attention Patching Scores',
        line=dict(color=colors['Attention'])
    ), row=1, col=2)
    
    # Construct title
    main_title = f'SFC error nodes {position_desc}: AtP Scores and Patching Scores Across Layers'
    if title_suffix:
        main_title += f' {title_suffix}'
    
    # Update layout for better readability
    fig.update_layout(
        title=main_title,
        xaxis_title='Layer Number',
        yaxis_title='Values',
        legend_title='Metrics',
        template='plotly_white',
        showlegend=True
    )
    
    fig.show()


plot_scores(
    resid_sfc_scores, 
    mlp_sfc_scores, 
    attn_sfc_scores,
    resid_patching_scores, 
    mlp_patching_scores, 
    attn_patching_scores,
    token_position=2
)


plot_scores(
    resid_sfc_scores, 
    mlp_sfc_scores, 
    attn_sfc_scores,
    resid_patching_scores, 
    mlp_patching_scores, 
    attn_patching_scores,
    token_position=3
)


plot_scores(
    resid_sfc_scores, 
    mlp_sfc_scores, 
    attn_sfc_scores,
    resid_patching_scores, 
    mlp_patching_scores, 
    attn_patching_scores,
    token_position=-2
)


# ### Plotting the SFC vs act. patching correlations

import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

def plot_correlation(sfc_scores, patching_scores, token_position=None, node_type='Resid'):
    # Process the tensors based on token_position
    if token_position is not None:
        # Extract values at specific token position
        sfc_values = [tensor[token_position].float().cpu().numpy() for tensor in sfc_scores]
        patching_values = [tensor[token_position].float().cpu().numpy() for tensor in patching_scores]

        position_desc = f"at token position {token_position}"
    else:
        # Compute mean across all positions
        sfc_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in sfc_scores]
        patching_values = [tensor.float().mean(dim=0).cpu().numpy() for tensor in patching_scores]
       
        position_desc = "averaged across token positions"
        
    """Plots a scatter plot of sfc_values vs patching_values with regression line and correlation coefficient, labeling points with their layer index."""
    layer_indices = list(range(len(sfc_values)))
    
    # Compute correlation coefficient
    r_value, _ = stats.pearsonr(sfc_values, patching_values)
    
    # Fit a linear regression line
    slope, intercept, _, _, _ = stats.linregress(sfc_values, patching_values)
    regression_line = [slope * x + intercept for x in sfc_values]
    
    # Define colors dynamically
    scatter_color = "rgba(50, 100, 250, 0.8)"  # Blue with some transparency
    line_color = "rgba(200, 50, 50, 0.9)"  # Red with some transparency
    text_color = "rgba(50, 50, 50, 0.6)"  # Dark gray with transparency
    
    # Create the scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sfc_values, 
        y=patching_values, 
        mode='markers+text', 
        name='Data Points',
        marker=dict(color=scatter_color, size=8),
        text=[str(i) for i in layer_indices],  # Layer indices as labels
        textposition='top center',
        textfont=dict(size=9, color=text_color)
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=sfc_values, 
        y=regression_line, 
        mode='lines',
        name='Regression Line',
        line=dict(color=line_color, width=2)
    ))

    title = f'{node_type} Error nodes {position_desc}: SFC scores vs patching scores'
    # Update layout
    fig.update_layout(
        title=f"{title} (r={r_value:.2f})",
        xaxis_title='SFC Scores',
        yaxis_title='Patching Scores',
        template='plotly_white'
    )
    
    fig.show()


plot_correlation(resid_sfc_scores, resid_patching_scores, token_position=-2)


plot_correlation(mlp_sfc_scores, mlp_patching_scores, token_position=-2, node_type='MLP')


plot_correlation(attn_sfc_scores, attn_patching_scores, token_position=-2, node_type='Attention')


# Life is good! The correlation of our SFC scores with the "ground truth" patching scores is almost perfect in all cases

plot_correlation(resid_sfc_scores, resid_patching_scores, token_position=3)




