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


# # Setting up the SAEs

from classes.sfc_model import SFC_Gemma

RUN_WITH_SAES = True # we'll need to run the model with attached SAEs to store the mean acts of SAE latents and errors

# Determine the caching device, where we'll load our SAEs and compute the SFC scores
if RUN_WITH_SAES:
    caching_device = device 
else:
    caching_device = "cuda:3"


caching_device


# For replicating the SFC part from the paper I used my custom SFC_Gemma class. In short, it
# - Loads a Gemma model and its Gemma Scope SAEs (either attaching them to the model or not)
# - Provides interface methods to compute SFC scores (currently, only attr patching is supported) on an arbitrary dataset (that follows the format of my SFCDatasetLoader class from above)

EXPERIMENT = 'sva_rc_be'

clear_cache()
sfc_model = SFC_Gemma(model, params_count=PARAMS_COUNT, control_seq_len=CONTROL_SEQ_LEN, 
                      attach_saes=RUN_WITH_SAES, caching_device=caching_device,
                      data_dir=datapath, experiment_name=EXPERIMENT)
clear_cache()

# sfc_model.print_saes()
# sfc_model.model.cfg
# , sfc_model.saes[0].cfg.dtype


# # Computing Mean Activations

# Here we'll call the corresponding method from the CircuitEvaluator class. This class encapsulates the SFC circuit evaluation algorithms.

from classes.sfc_evaluator import CircuitEvaluator

evaluator = CircuitEvaluator(sfc_model)


batch_size = 900
total_batches = None

# Reset the hooks to avoid weird bugs
sfc_model.model.reset_hooks()
if RUN_WITH_SAES:
    sfc_model._reset_sae_hooks()

# Below we'll call the main interface method for computing the mean scores
mean_scores = evaluator.compute_mean_node_activations(clean_dataset, corrupted_dataset, 
                                                      batch_size=batch_size, total_batches=total_batches)
mean_scores.keys()


# Total amount of scores should be n_layers * len(['resid', 'mlp', 'attn']) * len(['sae_latent', 'sae_error'])
assert len(mean_scores) == sfc_model.model.cfg.n_layers * 3 * 2


# Shapes check
mean_scores['blocks.0.attn.hook_z.hook_sae_error'].shape, mean_scores['blocks.0.attn.hook_z.hook_sae_acts_post'].shape, \
mean_scores['blocks.0.hook_mlp_out.hook_sae_error'].shape, mean_scores['blocks.0.hook_mlp_out.hook_sae_acts_post'].shape, \
mean_scores['blocks.0.hook_resid_post.hook_sae_error'].shape, mean_scores['blocks.0.hook_resid_post.hook_sae_acts_post'].shape

# Only the first attn error should have a different shape (accounting for each head) - [pos, n_head, d_head]
# All other errors should be [pos, d_model]
# All latents (hook_sae_acts_post) should be [pos, d_sae]


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
EXPERIMENT = 'sva_rc_be'

sfc_scores = SFC_NodeScores(
    device=device,
    data_dir=datapath,
    experiment_name=EXPERIMENT,
    load_if_exists=True  # This will automatically load our computed scores
)

sfc_scores.node_scores['blocks.0.attn.hook_z.hook_sae_error'].device #  checking if device mapping works (it does) 


# ### Checking error norms

# One simple thing to check is whether resid error norms increase monotonically with the layer number. Based on the results of `stage_1/analyze_error_scores.ipynb` - they should, so let's see if this holds for mean error activations.

resid_error_mean_act = sfc_scores.select_node_scores(lambda x: 'hook_resid_post.hook_sae_error' in x, 'mean_act')
print(f'Returned {len(resid_error_mean_act.keys())} keys: {resid_error_mean_act.keys()}')


resid_error_mean_act['blocks.0.hook_resid_post.hook_sae_error'].shape


# Compute norms of mean error activations
resid_error_mean_norms = {key: score_tensor.norm(dim=-1) for key, score_tensor in resid_error_mean_act.items()}
resid_error_mean_norms['blocks.0.hook_resid_post.hook_sae_error'].shape


def plot_mean_norms(mean_norms, positions_to_plot=[-2], title='Norms of mean SAE errors'):
    """
    Plots the mean norms of SAE errors for each layer using Plotly.
    
    Args:
        mean_norms (dict): Dictionary with keys like 'blocks.{j}.hook_resid_post.hook_sae_error' 
                          and values as tensors of shape [num_positions]
        positions_to_plot (list): List of positions to plot (indices into the first dimension), 
                                 default is [-2] (second to last)
        title (str): Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    import plotly.graph_objects as go
    import re
    
    # Get the number of positions from the first tensor's shape
    first_key = next(iter(mean_norms))
    num_positions = mean_norms[first_key].shape[0]
    
    # Ensure positions_to_plot is a list
    if not isinstance(positions_to_plot, list):
        positions_to_plot = [positions_to_plot]
    
    # Create the plot
    fig = go.Figure()
    
    for position in positions_to_plot:
        # Convert negative position index to positive if needed
        pos = position if position >= 0 else num_positions + position
        
        # Extract layer indices and corresponding norm values for the specified position
        layer_indices = []
        norm_values = []
        
        for key, norm_tensor in mean_norms.items():
            # Use regex to extract the layer number from keys like 'blocks.{j}.hook_resid_post.hook_sae_error'
            match = re.search(r'blocks\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                # Get the norm value for the specified position
                norm_value = float(norm_tensor[pos].item())
                
                layer_indices.append(layer_idx)
                norm_values.append(norm_value)
        
        # Sort by layer index to ensure correct ordering
        sorted_data = sorted(zip(layer_indices, norm_values))
        sorted_layer_indices, sorted_norm_values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Add a trace for this position
        fig.add_trace(
            go.Scatter(
                x=sorted_layer_indices,
                y=sorted_norm_values,
                mode='lines+markers',
                name=f'Position {position}'
            )
        )
    
    # Set plot title and labels
    fig.update_layout(
        title=title,
        xaxis_title='Layer',
        yaxis_title='Norm Value',
        xaxis=dict(tickmode='linear'),
        template='plotly_white',
        legend_title="Position"
    )
    
    return fig


# Plot multiple positions
fig = plot_mean_norms(resid_error_mean_norms, positions_to_plot=[2, 3, -2])
fig.show()


# So the result is overall expected, but one interesting observation is that the norm of the mean errors seems to be the highest for error nodes at position #3 ('that' token in SVA dataset) where we don't see important error nodes at all in the late layers.
# 
# Also note that "norm of mean errors" (plotted above) is not the same as "mean of error norms". Intuitively the "norm of mean errors" should be dominated by the "mean of error norms" because of the fact that "mean error" is a center of mass of a bunch of vectors. So e.g. when you have a lot of high-norm vectors which are kind of all in different directions, their center of mass would be somewhere in between around 0, making its norm much smaller than norms of individual vectors. 

# For curious readers, here's the (Claude-generated but looking correct) proof

# $$
# \text{For any set of vectors } \{\vec{e}_i\}_{i=1}^n \text{ in a Euclidean space, the following inequality holds:}
# $$
# 
# $$
# \left\|\frac{1}{n}\sum_{i=1}^n \vec{e}_i\right\| \leq \frac{1}{n}\sum_{i=1}^n \|\vec{e}_i\|
# $$
# 
# $$
# \text{Let } \vec{\mu} = \frac{1}{n}\sum_{i=1}^n \vec{e}_i \text{ be the mean of the error vectors.}
# $$
# 
# $$
# \text{The Euclidean norm } \|\cdot\| \text{ is a convex function due to the triangle inequality and the properties}
# $$
# $$
# \text{of square root. By Jensen's inequality, for a convex function } f \text{ and a random variable } X\text{:}
# $$
# 
# $$
# f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
# $$
# 
# $$
# \text{Applying this to our vectors with } f = \|\cdot\| \text{ and treating the vectors as random variables with}
# $$
# $$
# \text{uniform probability } \frac{1}{n}\text{:}
# $$
# 
# $$
# \|\mathbb{E}[\vec{e}]\| \leq \mathbb{E}[\|\vec{e}\|]
# $$
# 
# $$
# \text{Which gives us:}
# $$
# 
# $$
# \left\|\frac{1}{n}\sum_{i=1}^n \vec{e}_i\right\| \leq \frac{1}{n}\sum_{i=1}^n \|\vec{e}_i\|
# $$
# 
# $$
# \text{Equality Condition: The equality holds if and only if all vectors } \vec{e}_i \text{ are collinear with}
# $$
# $$
# \text{the same orientation, i.e., } \vec{e}_i = c_i\vec{v} \text{ for some unit vector } \vec{v} \text{ and } c_i \geq 0 \text{ (or } c_i \leq 0 \text{ for all } i\text{).}
# $$
# 
# $$
# \text{For example, if } \vec{e}_i = b_i \cdot \vec{\alpha} \text{ where } \|\vec{\alpha}\| = 1 \text{ and all } b_i \text{ have the same sign:}
# $$
# 
# $$
# \begin{align}
# \|\mathbb{E}[\vec{e}]\| &= \|\mathbb{E}[b_i \cdot \vec{\alpha}]\| = \|\mathbb{E}[b_i] \cdot \vec{\alpha}\| = |\mathbb{E}[b_i]| \cdot \|\vec{\alpha}\| = \mathbb{E}[b_i] \\
# \mathbb{E}[\|\vec{e}\|] &= \mathbb{E}[\|b_i \cdot \vec{\alpha}\|] = \mathbb{E}[|b_i| \cdot \|\vec{\alpha}\|] = \mathbb{E}[|b_i|] = \mathbb{E}[b_i]
# \end{align}
# $$
# 
# $$
# \text{Therefore, } \|\mathbb{E}[\vec{e}]\| = \mathbb{E}[\|\vec{e}\|] \text{ in this special case.}
# $$



