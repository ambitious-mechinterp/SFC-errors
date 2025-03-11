from transformer_lens import ActivationCache
from sae_lens import SAE, HookedSAETransformer
import torch
import numpy as np
from tqdm.notebook import tqdm
import einops
from jaxtyping import Float, Int
from torch import Tensor
from enum import Enum
from functools import partial
from pathlib import Path
import gc

# Add import for SFC_NodeScores
from .sfc_node_scores import SFC_NodeScores, AttributionAggregation

# utility to clear variables out of the memory & and clearing cuda cache
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

### Utility enums ###
class NodeScoreType(Enum):
    ATTRIBUTION_PATCHING = 'ATP'
    INTEGRATED_GRADIENTS = 'INT_GRADS'

class AttributionPatching(Enum):
    NORMAL = 'TRADITIONAL' # Approximates the effect of patching activations from patched run to the clean run
    ZERO_ABLATION = 'ZERO_ABLATION' # Approximates the effect of zeroing out activations from the patched/clean run

### Utility functions ###
def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    assert clean_dataset is not None or corrupted_dataset is not None, 'At least one dataset must be provided.'
    return_values = []

    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])

    return return_values

### Main class ###
class SFC_Gemma():
    def __init__(self, model, attach_saes=True, params_count=9, control_seq_len=1, caching_device=None,
                sae_resid_release=None, sae_attn_release=None, sae_mlp_release=None,
                sae_attn_width='16k', sae_mlp_width='16k', first_16k_resid_layers=None,
                data_dir=None, experiment_name=None):
        """
        Initializes the SFC_Gemma - wrapper around Gemma-2 models that can automatically load canonical SAEs and do SFC with them.
        
        Added parameters:
            data_dir: Path to directory for saving/loading scores
            experiment_name: Name of the experiment for saving/loading scores
        """
        if sae_resid_release is None:
            sae_resid_release = f'gemma-scope-{params_count}b-pt-res-canonical'

        if sae_attn_release is None:
            sae_attn_release = f'gemma-scope-{params_count}b-pt-att-canonical'

        if sae_mlp_release is None:
            sae_mlp_release = f'gemma-scope-{params_count}b-pt-mlp-canonical'

        self.model = model
        self.cfg = model.cfg
        self.device = model.cfg.device
        self.caching_device = caching_device
        self.control_seq_len = control_seq_len
        self.data_dir = data_dir
        self.experiment_name = experiment_name

        if params_count == 9:
            self.model.set_use_attn_in(True)
        self.model.set_use_attn_result(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)

        self.n_layers = self.cfg.n_layers
        self.d_model = self.cfg.d_model

        width_to_d_sae = lambda width: {
            '131k': 131072,
            '16k': 16384
        }[width]

        self.attn_d_sae = width_to_d_sae(sae_attn_width)
        self.mlp_d_sae = width_to_d_sae(sae_mlp_width)

        if first_16k_resid_layers is None:
            first_16k_resid_layers = self.n_layers

        print(f'Using 16K SAEs for the first {first_16k_resid_layers} layers, the rest {self.n_layers - first_16k_resid_layers} layer(s) - 131k SAEs')
        resid_saes_widths = ['16k'] * first_16k_resid_layers + ['131k'] * (self.n_layers - first_16k_resid_layers)

        self.resid_d_sae = [width_to_d_sae(width) for width in resid_saes_widths]

        # Initialize dictionary to store SAEs by type: resid, attn, mlp
        self.saes_dict = {
            'resid': [],
            'attn': [],
            'mlp': []
        }

        # Load all SAEs into the dictionary
        self.saes_dict['resid'] = [
            self._load_sae(sae_resid_release, f'layer_{i}/width_{resid_saes_widths[i]}/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['attn'] = [
            self._load_sae(sae_attn_release, f'layer_{i}/width_{sae_attn_width}/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['mlp'] = [
            self._load_sae(sae_mlp_release, f'layer_{i}/width_{sae_mlp_width}/canonical') for i in range(self.n_layers)
        ]
        self.saes = self.saes_dict['resid'] + self.saes_dict['mlp'] + self.saes_dict['attn']
        
        # Initialize SFC_NodeScores with the same parameters
        self.sfc_node_scores = SFC_NodeScores(
            device=self.device,
            caching_device=self.caching_device,
            control_seq_len=self.control_seq_len,
            data_dir=self.data_dir,
            experiment_name=self.experiment_name
        )
        
        # Attach all SAEs
        if attach_saes:
            self.add_saes()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_sfc_scores_for_templatic_dataset(self, clean_dataset, patched_dataset, batch_size=50, total_batches=None,
                                                 score_type: NodeScoreType = NodeScoreType.ATTRIBUTION_PATCHING, run_without_saes=True,
                                                 save_scores=True):
        """
        Main interface method for computing AtP scores for given clean and patched datasets.
        The method can run the model in two modes: with and without SAEs.
        1. With SAEs: the model be run with the attached SAEs, so that the gradients are computed automatically using PyTorch autograd.
        2. Without SAEs: the model will be run without SAEs, so that the gradients are computed analytically based on the stored activations.
        IMPORTANT: this requires the activations to be stored on the separate device (caching_device) to avoid OOM errors.
        
        Added parameters:
            save_scores: Whether to save scores to data_dir/experiment_name
        """
        if run_without_saes:
            print('Running without SAEs, gradients and activations will be computed analytically.')

        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_clean_scores = []
        metrics_patched = []

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            metric_clean = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()
            metric_patched = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean()

            if score_type == NodeScoreType.ATTRIBUTION_PATCHING:
                metric_clean, cache_clean, grad_clean = self.run_with_cache(clean_prompts, clean_attn_mask, metric_clean, 
                                                                            run_without_saes=run_without_saes)

                metric_patched, cache_patched, _ = self.run_with_cache(corrupted_prompts, corrupted_attn_mask, metric_patched, 
                                                                       run_backward_pass=False, run_without_saes=run_without_saes)

                if i == 0:
                    # Initialize node scores using SFC_NodeScores
                    self.sfc_node_scores.initialize_node_scores(
                        cache_clean, 
                        run_without_saes=run_without_saes,
                        d_sae_lookup_fn=self.key_to_d_sae,
                        hook_name_to_sae_act_name_fn=self.hook_name_to_sae_act_name
                    )
                    
                # Use our helper methods to update node scores
                self._update_node_scores(grad_clean, cache_clean, total_batches, 
                                        cache_patched=cache_patched, attr_type=AttributionPatching.NORMAL, 
                                        run_without_saes=run_without_saes)

                del grad_clean
            elif score_type == NodeScoreType.INTEGRATED_GRADIENTS:
                raise NotImplementedError('Integrated gradients are not implemented yet.')

            del cache_clean, cache_patched
            clear_cache()

            metrics_clean_scores.append(metric_clean)
            metrics_patched.append(metric_patched)

        clean_metric = torch.tensor(metrics_clean_scores).mean().item()
        patched_metric = torch.tensor(metrics_patched).mean().item()
        
        # Save scores if requested
        if save_scores and self.data_dir is not None and self.experiment_name is not None:
            self.sfc_node_scores.save_scores(mode="sfc")

        return (
            clean_metric, patched_metric,
            self.sfc_node_scores.node_scores
        )
    
    def compute_act_patching_scores_for_errors(self, clean_dataset, patched_dataset, batch_size=50, total_batches=None,
                                               token_specific_error_types=None, token_positions=None, layers_to_patch=None):
        """
        Compute activation patching scores for SAE error terms.
        
        Parameters:
        -----------
        clean_dataset: dict
            Dictionary containing the clean prompts, answers, etc.
        patched_dataset: dict
            Dictionary containing the patched/corrupted prompts, answers, etc.
        batch_size: int
            Number of samples to process in each batch
        total_batches: int or None
            Total number of batches to process. If None, process all available data.
        token_specific_error_types: list or None
            List of error types ('resid', 'mlp', 'attn') for which to perform token-specific patching.
            If None, all error types use global patching.
        token_positions: list/range or None
            Specific token positions to patch. If None, patch all token positions.
        layers_to_patch: dict or None
            Dictionary mapping error types to lists of layers to patch.
            If None, patch all layers for all error types.
        
        Returns:
        --------
        dict
            Dictionary mapping error types to patching scores
        """
        # Set defaults
        if token_specific_error_types is None:
            token_specific_error_types = []  # Empty list means no token-specific patching
        
        # Calculate how many total samples to process
        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size
            if n_prompts % batch_size != 0:
                total_batches += 1
        
        # Set token positions to patch (default to all positions)
        if token_positions is None:
            token_positions = range(seq_len)
        elif isinstance(token_positions, int):
            token_positions = [token_positions]  # Convert single position to list
            
        # Initialize default layers to patch (all layers for all error types)
        if layers_to_patch is None:
            layers_to_patch = {
                'resid': list(range(self.n_layers)),
                'mlp': list(range(self.n_layers)),
                'attn': list(range(self.n_layers)),
            }

        # Utilities for getting hook names for activation errors and patching them
        def get_error_activation_name(error_type, error_layer):
            if error_type == 'resid':
                hook_name = 'hook_resid_post'
            elif error_type == 'mlp':
                hook_name = 'hook_mlp_out'
            elif error_type == 'attn':
                hook_name = 'attn.hook_z'
            else:
                raise ValueError(f'Unknown error type: {error_type}')

            return f'blocks.{error_layer}.{hook_name}.hook_sae_error'

        # Global patching hook - patches all positions
        def global_patching_hook(act, hook, corrupted_cache):
            try:
                act[:] = corrupted_cache[hook.name][:]
            except KeyError as e:
                raise KeyError(f"Activation {hook.name} not found in corrupted cache.") from e
            return act

        # Token-specific patching hook - patches only specified position
        def token_specific_patching_hook(act, hook, corrupted_cache, position):
            try:
                # Copy only at the specified position, keeping dimensions intact
                if 'hook_z' in hook.name:  # Handle attention's different dimensionality
                    act[:, position, :, :] = corrupted_cache[hook.name][:, position, :, :]
                else:
                    act[:, position, :] = corrupted_cache[hook.name][:, position, :]
            except KeyError as e:
                raise KeyError(f"Activation {hook.name} not found in corrupted cache.") from e
            return act

        # Initialize result tensors - always use (n_layers, seq_len) shape for consistency
        model = self.model
        all_normalized_logit_dif = {
            'resid': torch.zeros((model.cfg.n_layers, seq_len), device=model.cfg.device),
            'mlp': torch.zeros((model.cfg.n_layers, seq_len), device=model.cfg.device),
            'attn': torch.zeros((model.cfg.n_layers, seq_len), device=model.cfg.device)
        }

        # Loop over batches
        for i in range(0, prompts_to_process, batch_size):
            print(f'---\nSamples {i}/{prompts_to_process}\n---')
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
                clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            # Get the corrupted cache (i.e. cache for patched prompts) for all the error nodes
            corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_prompts, attention_mask=corrupted_attn_mask, 
                                                                     names_filter=lambda name: 'error' in name)
            # Get the clean logits (output of the model on the clean prompts)
            clean_logits = model(clean_prompts, attention_mask=clean_attn_mask)
            
            # Get the logit_diff - difference between incorrect and correct answers' logits - for clean and corrupted prompts
            clean_logit_diff = self.get_logit_diff(clean_logits, clean_answers=clean_answers,
                                            patched_answers=corrupted_answers,
                                            answer_pos=clean_answers_pos)
            corrupted_logit_diff = self.get_logit_diff(corrupted_logits, clean_answers=clean_answers,
                                                patched_answers=corrupted_answers,
                                                answer_pos=corrupted_answers_pos)            
            # Compute the logit_dif baseline, that we'll use in the denominator later to compute the patching effects
            normalized_logit_dif_denom = torch.where(corrupted_logit_diff - clean_logit_diff == 0, 
                                                    torch.tensor(1, device=clean_logits.device), corrupted_logit_diff - clean_logit_diff)

            # Start patching for each error type
            for error_type in layers_to_patch.keys():
                print(f'Computing patching effect for {error_type} errors...')
                
                # Get the layers to patch for this error type
                layers = layers_to_patch[error_type]
                
                # Check if this error type should use token-specific patching
                if error_type in token_specific_error_types:
                    print(f"  Using token-specific patching for {len(token_positions)} positions")
                    
                    # Loop over layers
                    for layer_idx, layer in enumerate(tqdm(layers)):
                        # Loop over selected token positions
                        for pos_idx, pos in enumerate(token_positions):
                            # Create a position-specific patching hook
                            pos_hook = lambda act, hook, cache=corrupted_cache, position=pos: token_specific_patching_hook(act, hook, cache, position)
                            
                            # Run with this position-specific hook
                            logits = model.run_with_hooks(clean_prompts, attention_mask=clean_attn_mask, fwd_hooks=[
                                (get_error_activation_name(error_type, layer), pos_hook)
                            ])
                            
                            # Compute the logit diff for our patching run
                            logit_diff = self.get_logit_diff(logits, clean_answers=clean_answers,
                                                           patched_answers=corrupted_answers,
                                                           answer_pos=clean_answers_pos)
                            
                            # Compute normalized logit diff
                            normalized_logit_dif = (logit_diff - clean_logit_diff) / normalized_logit_dif_denom
                            
                            # Store the result for this specific position
                            all_normalized_logit_dif[error_type][layer, pos] += normalized_logit_dif.mean(0)
                            
                            del logits, logit_diff, normalized_logit_dif
                            clear_cache()
                else:
                    print(f"  Using global patching")
                    
                    # Create a global patching hook
                    global_hook = lambda act, hook, cache=corrupted_cache: global_patching_hook(act, hook, cache)
                    
                    # Loop over layers
                    for layer_idx, layer in enumerate(tqdm(layers)):
                        # Apply global patching
                        logits = model.run_with_hooks(clean_prompts, attention_mask=clean_attn_mask, fwd_hooks=[
                            (get_error_activation_name(error_type, layer), global_hook)
                        ])
                        
                        # Compute the logit diff for our patching run
                        logit_diff = self.get_logit_diff(logits, clean_answers=clean_answers,
                                                       patched_answers=corrupted_answers,
                                                       answer_pos=clean_answers_pos)
                        
                        # Compute normalized logit diff
                        normalized_logit_dif = (logit_diff - clean_logit_diff) / normalized_logit_dif_denom
                        
                        # Store the result for global patching - broadcast to all token positions for this layer
                        mean_effect = normalized_logit_dif.mean(0).item()
                        all_normalized_logit_dif[error_type][layer, :] += mean_effect
                        
                        del logits, logit_diff, normalized_logit_dif
                        clear_cache()
                        
            del corrupted_cache, clean_logits, corrupted_logits, clean_logit_diff, corrupted_logit_diff, normalized_logit_dif_denom
            clear_cache()
        
        # Divide by total_batches to get the average
        for error_type in all_normalized_logit_dif.keys():
            all_normalized_logit_dif[error_type] /= total_batches

        return all_normalized_logit_dif

    def compute_activation_analysis(self, clean_dataset, patched_dataset, activation_types=['sae_error'], batch_size=50, total_batches=None):
        """
        Computes analysis of model activations by returning:
        1. Norm of activation difference (patched - clean)
        2. Norm of gradient for each activation
        3. Attribution patching score (dot product of the above)
        
        This allows for visualization to understand the relationship between activation
        differences and gradients in model components.
        
        Args:
            clean_dataset: Dataset with clean prompts
            patched_dataset: Dataset with patched prompts
            activation_types: List of activation types to analyze. Default is ['sae_error']
            batch_size: Number of samples to process in each batch
            total_batches: Total number of batches to process (None means process all)
        
        Returns:
            dict: Dictionary with keys corresponding to activation points and values being tuples of
                (activation_diff_norm, gradient_norm, atp_score)
        """
        # Ensure SAEs are attached
        if not self.are_saes_attached():
            raise ValueError("This method requires SAEs to be attached. Call add_saes() first.")
        
        # Define activation types to analyze
        if activation_types is None:
            activation_types = ['sae_error', 'resid_post']
        
        # Calculate how many total samples to process
        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]
        
        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches
        
        if total_batches is None:
            total_batches = n_prompts // batch_size
            if n_prompts % batch_size != 0:
                total_batches += 1
        
        # Initialize data structures to store results
        activation_data = {}
        
        # Define filters for forward and backward hooks based on activation types
        fwd_filters = []
        bwd_filters = []
        
        if 'sae_error' in activation_types:
            fwd_filters.append(lambda name: 'hook_sae_error' in name)
        
        if 'resid_post' in activation_types:
            fwd_filters.append(lambda name: name.endswith('hook_resid_post.hook_sae_input'))

        # 'hook_sae_output' and 'hook_sae_input' are used in both cases to ensure correct gradient computation with SAEs attached
        bwd_filters.append(lambda name: 'hook_sae_output' in name)
        bwd_filters.append(lambda name: 'hook_sae_input' in name)
        
        # Combine all filters
        fwd_cache_filter = lambda name: any(f(name) for f in fwd_filters)
        bwd_cache_filter = lambda name: any(f(name) for f in bwd_filters)
        
        # Process each batch
        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)
            
            # Define metrics for clean and patched runs
            metric_clean = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()
            
            # Run clean model with cache to get gradients and activations
            metric_clean_val, cache_clean, grad_clean = self.run_with_cache(
                clean_prompts, 
                clean_attn_mask, 
                metric_clean,
                fwd_cache_filter=fwd_cache_filter,
                bwd_cache_filter=bwd_cache_filter,
                bwd_activations_to_cache=activation_types,
                run_backward_pass=True,
                run_without_saes=False
            )
            
            # Run patched model with cache to get activations (no gradients needed)
            metric_patched_val, cache_patched, _ = self.run_with_cache(
                corrupted_prompts,
                corrupted_attn_mask,
                lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean(),
                fwd_cache_filter=fwd_cache_filter,
                run_backward_pass=False,
                run_without_saes=False
            )
            
            # Initialize data structure for this batch
            batch_data = {}
            
            # Process each activation point
            for key in cache_clean.keys():
                # Get activation difference between patched and clean
                activation_diff = cache_patched[key] - cache_clean[key]
                
                # Check if gradient exists for this key
                if key not in grad_clean:
                    print(f"Warning: No gradient found for {key}. Available keys: {list(grad_clean.keys())[:5]}...")
                    continue
                
                gradient = grad_clean[key]
                
                # Reshape the attention hook activations & gradients to flatten the n_head dimension if needed
                if 'hook_z' in key:
                    activation_diff = einops.rearrange(activation_diff, 
                                                    'batch pos n_head d_head -> batch pos (n_head d_head)')
                    gradient = einops.rearrange(gradient,
                                            'batch pos n_head d_head -> batch pos (n_head d_head)')
                
                # Compute our analysis metrics: norms of activation difference and gradient, and AtP score
                activation_diff_norm = torch.norm(activation_diff, dim=-1)  # Norm across model dimension
                gradient_norm = torch.norm(gradient, dim=-1)  # Norm across model dimension
                atp_score = einops.einsum(gradient, activation_diff, 'batch pos d_act, batch pos d_act -> batch pos')
                
                # Store results for this activation point in this batch
                if key not in batch_data:
                    batch_data[key] = []
                
                # Store as tuple (activation_diff_norm, gradient_norm, atp_score)
                batch_data[key].append((activation_diff_norm, gradient_norm, atp_score))

                # del activation_diff, gradient
                # clear_cache()
            
            # Update main data structure with batch results
            for key, batch_tuples in batch_data.items():
                if key not in activation_data:
                    activation_data[key] = []
                activation_data[key].extend(batch_tuples)
            
            # Clean up to prevent memory issues
            del cache_clean, cache_patched, grad_clean, batch_data
            clear_cache()
        
        # Aggregate results across batches
        aggregated_data = {}
        for key, tuples_list in activation_data.items():
            # Concatenate all batch data
            activation_diff_norms = torch.cat([t[0] for t in tuples_list], dim=0)
            gradient_norms = torch.cat([t[1] for t in tuples_list], dim=0)
            atp_scores = torch.cat([t[2] for t in tuples_list], dim=0)
            
            # Average across batch dimension
            activation_diff_norms_mean = activation_diff_norms.mean(dim=0)  # Shape: [pos]
            gradient_norms_mean = gradient_norms.mean(dim=0)  # Shape: [pos]
            atp_scores_mean = atp_scores.mean(dim=0)  # Shape: [pos]
            
            # Store aggregated results
            aggregated_data[key] = (activation_diff_norms_mean, gradient_norms_mean, atp_scores_mean)
        
        return aggregated_data

    def run_with_cache(self, tokens: Int[Tensor, "batch pos"],
                       attn_mask: Int[Tensor, "batch pos"], metric,
                       fwd_cache_filter=None, bwd_cache_filter=None, bwd_activations_to_cache=['sae_error'], 
                       run_backward_pass=True, run_without_saes=False):
        """
        Runs the model on the given tokens and stores the relevant forward and backward caches.
        The default behavior varies depending on the run_without_saes flag:
        1. If run_without_saes is True, the model runs without SAEs and caches the forward activations and gradients, 
        so that the gradients w.r.t. SAE latents and error terms can be computed analytically.
        2. If run_without_saes is False, the model runs with SAEs attached and computes the gradients using PyTorch autograd.
        This is performed inside the _set_backward_hooks method.
        """
        if run_without_saes:
            if self.are_saes_attached():
                raise ValueError('Forward & Backward passes are performed analytically, but SAEs are still attached. Call reset_saes() first to save VRAM.')
            if self.caching_device is None:
                raise ValueError('Caching device must be provided for no-SAEs runs.')
        
        if fwd_cache_filter is None:
            if run_without_saes:
                fwd_cache_filter = lambda name: 'resid_post' in name or 'attn.hook_z' in name or 'mlp_out' in name
            else:
                # Take the SAE latents and error term activations by default
                fwd_cache_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_error' in name

        cache = {}
        def forward_cache_hook(act, hook):
            cache[hook.name] = act.detach().to(self.caching_device)

        self.model.add_hook(fwd_cache_filter, forward_cache_hook, "fwd")

        grad_cache = {}

        try:
            if run_backward_pass:
                self._set_backward_hooks(grad_cache, bwd_cache_filter, compute_grad_analytically=run_without_saes, 
                                         activations_to_cache=bwd_activations_to_cache)

                # Enable gradients only during the backward pass
                with torch.set_grad_enabled(True):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
                    metric_value.backward()  # Compute gradients
            else:
                # Forward pass only
                with torch.set_grad_enabled(False):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
        finally:
            clear_cache()
            # Ensure hooks are reset regardless of exceptions
            self.model.reset_hooks()
            if self.are_saes_attached():
                self._reset_sae_hooks()

        return (
            metric_value.item(),
            ActivationCache(cache, self.model).to(self.caching_device),
            ActivationCache(grad_cache, self.model).to(self.caching_device),
        )

    def _set_backward_hooks(self, grad_cache, bwd_hook_filter=None, compute_grad_analytically=False, 
                            activations_to_cache=['sae_error']):
        if bwd_hook_filter is None:
            if compute_grad_analytically:
                bwd_hook_filter = lambda name: 'resid_post' in name or 'attn.hook_z' in name or 'mlp_out' in name
            else:
                bwd_hook_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_output' in name or 'hook_sae_input' in name

        temp_cache = {}

        if compute_grad_analytically:
            # Just store the gradients in the analytical case, we'll use them later
            def backward_cache_hook(gradient, hook):
                grad_cache[hook.name] = gradient.detach().to(self.caching_device)
        else:
            # Computing grads non-analytically using Pytorch autograd
            def backward_cache_hook(gradient, hook):
                if 'hook_sae_output' in hook.name:
                    hook_sae_error_name = hook.name.replace('hook_sae_output', 'hook_sae_error')

                    # Optionally store the gradients for the SAE error terms
                    if any([act_name in hook_sae_error_name for act_name in activations_to_cache]):
                        grad_cache[hook_sae_error_name] = gradient.detach()

                    # We're storing the gradients for the SAE output activations to copy them to the SAE input activations gradients
                    if not 'hook_z' in hook.name:
                        temp_cache[hook.name] = gradient.detach()
                    else: # In the case of attention hook_z hooks, reshape them to match the SAE input shape, which doesn't include n_heads
                        hook_z_grad = einops.rearrange(gradient.detach(),
                                                    'batch pos n_head d_head -> batch pos (n_head d_head)')
                        temp_cache[hook.name] = hook_z_grad
                elif 'hook_sae_input' in hook.name:
                    # We're copying the gradients from the SAE output activations to the SAE input activations gradients
                    sae_output_grad_name = hook.name.replace('hook_sae_input', 'hook_sae_output')

                    gradient = temp_cache[sae_output_grad_name] # this ensures that gradient propagation is unaffected by SAEs

                    # Optionally store the gradients for the SAE input activations
                    if any([act_name in hook.name for act_name in activations_to_cache]):
                        grad_cache[hook.name] = gradient.detach()

                    # Pass-through: use the downstream gradients
                    return (gradient,)
                else:
                    # Default case (SAE latents): just store the gradients
                    grad_cache[hook.name] = gradient.detach()

        self.model.add_hook(bwd_hook_filter, backward_cache_hook, "bwd")
    
    def _update_node_scores(self, grad_cache, cache_clean, total_batches,
                           cache_patched=None, attr_type=AttributionPatching.NORMAL, 
                           run_without_saes=True, batch_reduce='mean'):
        """
        Helper method that delegates node score updates to the appropriate method 
        based on whether SAEs are attached.
        """
        if attr_type.value == AttributionPatching.NORMAL.value:
            assert cache_patched is not None, 'Patched cache must be provided for normal attribution patching.'
            
        if not run_without_saes:
            self._update_node_scores_saes_run(grad_cache, cache_clean, total_batches, 
                                             cache_patched=cache_patched, attr_type=attr_type, 
                                             batch_reduce=batch_reduce)
        else:
            self._update_node_scores_no_saes_run(grad_cache, cache_clean, total_batches, 
                                               cache_patched=cache_patched, attr_type=attr_type, 
                                               batch_reduce=batch_reduce)
    
    def _update_node_scores_no_saes_run(self, clean_grad, cache_clean, total_batches,
                                       cache_patched=None, attr_type=AttributionPatching.NORMAL, 
                                       batch_reduce='mean'):
        """
        Method that computes the AtP scores for the current batch, assuming that the model is run without SAEs attached.
        This means that we'll FIRST need to compute the SAE latents and error terms activations AND gradients, and then
        compute the scores based on them.
        """
        node_scores = self.sfc_node_scores.node_scores

        def compute_score_update(key):
            """
            Given the activation name `key`, which can be either resid_post, attn.hook_z or mlp_out,
            computes the corresponding SAE latent activations and error terms, and then computes the AtP scores based on them.
            """
            # Get the activations that are input to the SAE
            clean_acts = cache_clean[key]
            current_grads = clean_grad[key]
            if attr_type.value == AttributionPatching.NORMAL.value:
                patched_acts = cache_patched[key]

            # Step-1: Compute the SAE latents and error terms
            sae = self.get_sae_by_hook_name(key)

            sae_latents_act_clean = sae.encode(clean_acts)
            sae_out_clean = sae.decode(sae_latents_act_clean)
            sae_error_clean = clean_acts - sae_out_clean

            if attr_type.value == AttributionPatching.NORMAL.value:
                sae_latents_act_patched = sae.encode(patched_acts)
                sae_out_patched = sae.decode(sae_latents_act_patched)
                sae_error_patched = patched_acts - sae_out_patched

            # Step-2: Compute the gradients w.r.t. the SAE latents and error terms            
            if 'hook_z' in key: # Reshape the attn hook_z gradients to flatten the n_head dimension
                current_grads = einops.rearrange(current_grads, 
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')
                sae_error_clean = einops.rearrange(sae_error_clean,
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')
                if attr_type.value == AttributionPatching.NORMAL.value:
                    sae_error_patched = einops.rearrange(sae_error_patched,
                                                'batch pos n_head d_head -> batch pos (n_head d_head)')

            sae_latent_grad = einops.einsum(current_grads, sae.W_dec,
                                            'batch pos d_act, d_sae d_act -> batch pos d_sae')
            sae_error_grad = current_grads # shape [batch pos d_act]

            # Step-3 (final): Compute the score update
            if attr_type.value == AttributionPatching.NORMAL.value:
                activation_term_latents = sae_latents_act_patched - sae_latents_act_clean
                activation_term_error = sae_error_patched - sae_error_clean
            else:
                activation_term_latents = -sae_latents_act_clean
                activation_term_error = -sae_error_clean

            # SAE error term case: we want a single score per error term, so we're multiplying the d_act dimension out
            error_score_update = einops.einsum(sae_error_grad, activation_term_error,
                                               'batch pos d_act, batch pos d_act -> batch pos')
            
            # SAE latents case: we want a score per each feature, so we're keeping the d_sae dimension
            latents_score_update = sae_latent_grad * (activation_term_latents) # shape [batch pos d_sae]

            return latents_score_update, error_score_update
        # end of compute_score_update

        # Now we just loop over the activations (resid, attn, mlp) and compute the scores of each node corresponding to them,
        # calling the compute_score_update() from above
        for key in cache_clean.keys():
            latents_score_update, error_score_update = compute_score_update(key)
            sae_acts_post_name, sae_error_name = self.hook_name_to_sae_act_name(key)

            if batch_reduce == 'sum':
                latents_score_update = latents_score_update.sum(0)
                error_score_update = error_score_update.sum(0)

                node_scores[sae_acts_post_name] += latents_score_update
                node_scores[sae_error_name] += error_score_update
            elif batch_reduce == 'mean':
                latents_score_update = latents_score_update.mean(0)
                error_score_update = error_score_update.mean(0)

                node_scores[sae_acts_post_name] += latents_score_update / total_batches
                node_scores[sae_error_name] += error_score_update / total_batches

    def _update_node_scores_saes_run(self, clean_grad, cache_clean, total_batches, 
                               cache_patched=None, batch_reduce='mean', attr_type=AttributionPatching.NORMAL):
        """
        Computes the attribution patching scores for the current batch, assuming that the model is run with SAEs attached.
        This means that we just need to multiply the gradients w.r.t. the SAE latents and error terms with the activation terms.
        """
        node_scores = self.sfc_node_scores.node_scores
        
        for key in node_scores.keys():
            if attr_type.value == AttributionPatching.NORMAL.value:
                activation_term = cache_patched[key] - cache_clean[key]
            elif attr_type.value == AttributionPatching.ZERO_ABLATION.value:
                # In the zero ablation variant, we set patched activations to zero
                activation_term = -cache_clean[key]

            if 'hook_sae_error' in key:
                # SAE error term case: we want a single score per error term,
                # so we're multiplying the d_act dimension out
                if 'hook_z.hook_sae_error' not in key:
                    score_update = einops.einsum(clean_grad[key], activation_term,
                                                'batch pos d_act, batch pos d_act -> batch pos')
                else:
                    score_update = einops.einsum(clean_grad[key], activation_term,
                                                'batch pos n_head d_head, batch pos n_head d_head -> batch pos')
            else:
                # SAE latents case: we want a score per each feature, so we're keeping the d_sae dimension
                score_update = clean_grad[key] * (activation_term) # shape [batch pos d_sae]

            if batch_reduce == 'sum':
                score_update = score_update.sum(0)
                node_scores[key] += score_update
            elif batch_reduce == 'mean':
                score_update = score_update.mean(0)
                node_scores[key] += score_update / total_batches

    def get_answer_logit(self, logits: Float[Tensor, "batch pos d_vocab"], clean_answers: Int[Tensor, "batch"],
                         ansnwer_pos: Int[Tensor, "batch"], return_all_logits=False) -> Float[Tensor, "batch"]:
        # clean_answers_pos_idx = clean_answers_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, logits.size(1), logits.size(2))

        answer_pos_idx = einops.repeat(ansnwer_pos, 'batch -> batch 1 d_vocab',
                                       d_vocab=logits.shape[-1])
        answer_logits = logits.gather(1, answer_pos_idx).squeeze(1) # shape [batch, d_vocab]

        correct_logits = answer_logits.gather(1, clean_answers.unsqueeze(1)).squeeze(1) # shape [batch]

        if return_all_logits:
            return answer_logits, correct_logits

        return correct_logits

    def get_logit_diff(self, logits: Float[Tensor, "batch pos d_vocab"],
                    clean_answers: Int[Tensor, "batch"], patched_answers: Int[Tensor, "batch count"],
                    answer_pos: Int[Tensor, "batch"], patch_answer_reduce='max') -> Float[Tensor, "batch"]:
        """
        Computes the standard SFC metric: logit difference between the incorrect answers and the correct answer.
        The method extracts the log probabilities for the correct answer and the incorrect answers from the `logits` tensor,
            using the answer_pos indices indicating at which token to look for the answer,
            and clean_answers and patched_answers tensors indicating what are the tokens for correct and incorrect answers, respectively.
        """
        # Compute the logits for the correct answers and the tokens they have been computed at (answer_logits)
        answer_logits, correct_logits = self.get_answer_logit(logits, clean_answers, answer_pos, return_all_logits=True)

        if patched_answers.dim() == 1:  # If there's only one incorrect answer, gather the incorrect answer logits
            incorrect_logits = answer_logits.gather(1, patched_answers.unsqueeze(1)).squeeze(1)  # shape [batch]
        else:
            incorrect_logits = answer_logits.gather(1, patched_answers)  # shape [batch, answer_count]

        # If there are multiple incorrect answer options, incorrect_logits is now of shape [batch, answer_count]
        if patched_answers.dim() == 2:
            # Sum the logits for each incorrect answer option
            if patch_answer_reduce == 'sum':
                incorrect_logits = incorrect_logits.sum(dim=1)
            # Or take their maximum: this should be a better option to avoid situations where the model outputs gibberish and all the answers have similar logits
            elif patch_answer_reduce == 'max':
                incorrect_logits = incorrect_logits.max(dim=1).values

        # Both logit tensors are now of shape [batch]
        return incorrect_logits - correct_logits

    def get_sae_by_hook_name(self, hook_name):
        layer_num = self.hook_name_to_layer_number(hook_name)

        if 'attn.hook_z' in hook_name:
            return self.saes_dict['attn'][layer_num]
        elif 'hook_mlp_out' in hook_name:
            return self.saes_dict['mlp'][layer_num]
        elif 'hook_resid_post' in hook_name:
            return self.saes_dict['resid'][layer_num]
        else:
            raise ValueError(f'Invalid hook name: {hook_name}')
    
    def hook_name_to_sae_act_name(self, hook_name):
        # Split the input string by periods
        parts = hook_name.split('.')
        
        # Validate the input format
        if len(parts) < 3 or parts[0] != 'blocks':
            raise ValueError("Input string must start with 'blocks.<index>.'")

        # Extract the index and hook_name
        index = parts[1]
        hook_name = '.'.join(parts[2:])  # Handles cases where hook_name contains dots

        # Construct the desired SAE names
        sae_acts_post = f"blocks.{index}.{hook_name}.hook_sae_acts_post"
        sae_error = f"blocks.{index}.{hook_name}.hook_sae_error"

        return sae_acts_post, sae_error
    
    def _sae_act_name_to_hook_name(self, sae_act_name):
        """
        Convert a SAE activation name (hook_sae_acts_post or hook_sae_error) back to the original hook name.
        This is the inverse of hook_name_to_sae_act_name.
        """
        # Split the input string by periods
        parts = sae_act_name.split('.')
        
        # Validate the input format
        if len(parts) < 3 or parts[0] != 'blocks':
            raise ValueError("Input string must start with 'blocks.<index>.'")
            
        # Extract the index and hook_name without the SAE suffix
        index = parts[1]
        hook_name_parts = []
        
        # Extract the hook name parts before ".hook_sae_acts_post" or ".hook_sae_error"
        for part in parts[2:]:
            if 'hook_sae_acts_post' in part:
                hook_name_parts.append(part.split('.hook_sae_acts_post')[0])
                break
            elif 'hook_sae_error' in part:
                hook_name_parts.append(part.split('.hook_sae_error')[0])
                break
            else:
                hook_name_parts.append(part)
        
        # Construct the original hook name
        original_hook_name = f"blocks.{index}.{'.'.join(hook_name_parts)}"
        return original_hook_name
    
    def hook_name_to_layer_number(self, hook_name):
        # Split the input string by periods
        parts = hook_name.split('.')
        
        # Validate that the string has the correct format
        if len(parts) < 3 or parts[0] != 'blocks':
            raise ValueError("Input string must start with 'blocks.<index>.'")
        
        # Extract and return the block number as an integer
        return int(parts[1])

    def key_to_d_sae(self, key):
        if 'resid' in key:
            layer_num = self.hook_name_to_layer_number(key)
            return self.resid_d_sae[layer_num]
        elif 'attn' in key:
            return self.attn_d_sae
        elif 'mlp' in key:
            return self.mlp_d_sae

    def reset_saes(self):
        self.model.reset_saes()

    def are_saes_attached(self):
        return bool(self.model.acts_to_saes)

    def add_saes(self):
        for sae in self.saes:
            self.model.add_sae(sae, use_error_term=True)

    def print_saes(self):
        if not self.are_saes_attached():
            print('SAEs are not attached to the model.')
            return
        
        saes = self.model.acts_to_saes
        print(f'Number of SAEs: {len(saes)}')

        for name, sae in saes.items():
            print(name, sae)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _reset_sae_hooks(self):
        for sae in self.saes:
            sae.reset_hooks()

    def _load_sae(self, sae_release, sae_id):
        sae_device = self.device if self.caching_device is None else self.caching_device

        return SAE.from_pretrained(sae_release, sae_id, device=sae_device)[0].to(torch.bfloat16)