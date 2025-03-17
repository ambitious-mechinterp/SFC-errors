import torch
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
import gc
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
from jaxtyping import Float, Int
from torch import Tensor
import einops

# Import SFC_NodeScores
from .sfc_node_scores import SFC_NodeScores
from .sfc_model import sample_dataset

# Utility to clear variables out of the memory & clearing cuda cache
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

class CircuitEvaluator:
    """
    Class for evaluating circuit faithfulness.
    """
    def __init__(self, model_wrapper, sfc_node_scores: SFC_NodeScores = None):
        """
        Initialize the CircuitEvaluator.
        
        Args:
            model_wrapper: A wrapper around the model (e.g., SFC_Gemma)
            sfc_node_scores: SFC_NodeScores instance with pre-computed scores
        """
        self.model_wrapper = model_wrapper
        self.sfc_node_scores = sfc_node_scores if sfc_node_scores is not None else model_wrapper.sfc_node_scores
        
        # Store model and device information for convenience
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self.control_seq_len = self.sfc_node_scores.control_seq_len
        
        # Check if we have the required scores
        if self.sfc_node_scores.node_scores is None:
            print("No SFC scores found. You may need to compute them first.")
            
        if self.sfc_node_scores.mean_activations is None:
            print("No mean activations found. You may need to compute them with compute_mean_node_activations.")
        
    def compute_mean_node_activations(self, clean_dataset, corrupted_dataset=None, batch_size=50, total_batches=None, 
                                      save_activations=True):
        """
        Computes mean activations for each node in the model.
        This is used for later calculating faithfulness scores.
        
        Args:
            clean_dataset: Clean dataset to compute mean activations on (required)
            corrupted_dataset: Optional corrupted dataset to also include in the mean computation
            batch_size: Batch size for processing
            total_batches: Total number of batches to process (defaults to all)
            save_activations: Whether to save activations to data_dir/experiment_name
            
        Returns:
            Dictionary of mean activations
        """
        # Check if SAEs are attached
        if not self.model_wrapper.are_saes_attached():
            print('SAEs not attached. Attaching SAEs for mean activation computation.')
            self.model_wrapper.add_saes()

        # Determine whether we're processing one or two datasets
        datasets = []
        if clean_dataset is not None:
            datasets.append(clean_dataset)
        if corrupted_dataset is not None:
            datasets.append(corrupted_dataset)
            
        if len(datasets) == 0:
            raise ValueError("At least clean_dataset must be provided")
        
        # Count total batches to process for normalization
        total_batches_to_process = 0
        for dataset in datasets:
            n_prompts = dataset['prompt'].shape[0]
            dataset_batches = n_prompts // batch_size
            if n_prompts % batch_size != 0:
                dataset_batches += 1
                
            if total_batches is not None:
                dataset_batches = min(dataset_batches, total_batches)
                
            total_batches_to_process += dataset_batches
        
        # Create a filter for the forward hooks to capture SAE activations
        fwd_cache_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_error' in name
        
        # Process each dataset
        mean_activations = None
        batches_processed = 0
        
        for dataset_idx, dataset in enumerate(datasets):
            n_prompts, seq_len = dataset['prompt'].shape
            
            prompts_to_process = n_prompts if total_batches is None else min(n_prompts, batch_size * total_batches)
            
            print(f"Processing {'clean' if dataset_idx == 0 else 'corrupted'} dataset:")
            for i in tqdm(range(0, prompts_to_process, batch_size)):
                # Extract batch data
                prompts = dataset['prompt'][i:i+batch_size]
                attention_mask = dataset['attention_mask'][i:i+batch_size] if 'attention_mask' in dataset else None
                
                # Run model with cache using transformer_lens model directly
                with torch.no_grad():
                    # Use transformer_lens's run_with_cache method
                    output, cache = self.model.run_with_cache(
                        prompts, 
                        attention_mask=attention_mask,
                        return_type=None,
                        remove_batch_dim=False,
                        names_filter=fwd_cache_filter
                    )
                
                # On first batch of first dataset, initialize mean_activations structure
                if mean_activations is None:
                    self.sfc_node_scores.initialize_mean_activations(cache, run_without_saes=False)
                    mean_activations = self.sfc_node_scores.mean_activations
                
                # Increment batches processed counter
                batches_processed += 1
                
                # For each node, accumulate activations
                for key in mean_activations.keys():
                    if key in cache.keys():
                        acts = cache[key]
                        
                        # Simply accumulate the mean across batch dimension
                        # We preserve the original tensor shape (including head structure for attention)
                        mean_activations[key] += acts.mean(0) / total_batches_to_process
                
                # Clear cache to free memory
                del cache, output
                clear_cache()
        
        # Store in SFC_NodeScores object
        self.sfc_node_scores.mean_activations = mean_activations
        
        # Save activations if requested
        if save_activations and self.sfc_node_scores.data_dir is not None and self.sfc_node_scores.experiment_name is not None:
            self.sfc_node_scores.save_scores(mode="mean_act")
            
        return mean_activations
    
    def determine_nodes_to_ablate(self, node_threshold: float) -> Dict[str, torch.Tensor]:
        """
        Get binary masks for each node indicating which elements are below the threshold, 
        i.e. nodes to ablate when running circuit faithfulness/completeness evaluation.

        In SFC paper notations, returns the binary masks for the set of nodes in M \ C,
            where M is the full set of nodes and C is the circuit nodes.
        
        Args:
            node_threshold: Threshold to determine whether elements are included in the circuit
            
        Returns:
            Dictionary mapping node names to binary masks where 1 indicates the element is to be ablated (not part of the circuit).
            The masks are of shape [seq_len] for error nodes and [seq_len, d_sae] for SAE latent nodes.
        """
        if self.sfc_node_scores.node_scores is None:
            raise ValueError("SFC scores not computed. Compute SFC scores first.")
        
        # Dictionary to store binary masks
        nodes_ablation_mask = {}
        
        for key, scores in self.sfc_node_scores.node_scores.items():
            # For feature nodes it's a 2D mask (pos, feature_idx), for error nodes it's a 1D mask (pos)
            mask = (scores < node_threshold).to(device=self.device)
            
            nodes_ablation_mask[key] = mask
                
        return nodes_ablation_mask
    
    def evaluate_circuit_faithfulness(self, clean_dataset, patched_dataset, node_threshold: float, 
                                      always_ablate_fn=lambda act_name: False,
                                      cutoff_early_layers=True,
                                      nodes_to_restore: Optional[List[str]] = None,
                                      correct_for_the_nodes_to_restore: bool = False,
                                      batch_size=100, total_batches=None,
                                      verbose=True) -> Tuple[torch.Tensor, int]:
        """
        Evaluate the faithfulness of a circuit defined by nodes above threshold.
        
        Args:
            clean_dataset: The clean dataset to evaluate on
            patched_dataset: The patched dataset to evaluate on
            node_threshold: Threshold to determine circuit nodes
            always_ablate_fn: Function that determines which nodes should always be ablated
                              regardless of threshold. Defaults to a function that returns False (no nodes are always ablated).
            nodes_to_restore: List of node names to restore activation for (i.e. substitute their values as they would be without ablation)
                Defaults to an empty list
            correct_for_the_nodes_to_restore: Whether to correct the number of nodes in the circuit for the nodes to restore
                Defaults to False (for easier comparison between runs)
            batch_size: Number of examples to process at once
            total_batches: Total number of batches to process
            verbose: Whether to print progress information
            
        Returns:
            - Faithfulness metrics tensor of shape [total_batches * 2] for each batch (and clean & patched dataset),
            - Total number of nodes in the circuit
        """
        # Count total batches and prompts to process
        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size
            if n_prompts % batch_size != 0:
                total_batches += 1
        
        nodes_ablation_mask = self.determine_nodes_to_ablate(node_threshold)

        # Check if we have the mean activations pre-computed
        if self.sfc_node_scores.mean_activations is None:
            raise ValueError("Mean activations not computed. Compute mean activations first.")

        if nodes_to_restore is None:
            nodes_to_restore = []
        
        # Check if the mean activations keys match the sfc_node_scores keys
        if set(nodes_ablation_mask.keys()) != set(self.sfc_node_scores.mean_activations.keys()):
            raise ValueError("Mismatch between circuit masks and mean activations keys.")

        # Optimization: Initialize empty circuit metrics cache if it doesn't exist
        if not hasattr(self, '_empty_circuit_metrics_cache'):
            self._empty_circuit_metrics_cache = {}
        
        # Create a cache key based on dataset properties and other parameters that might affect the metric
        # This is needed so that we don't need to recompute the empty circuit metric for the same dataset and parameters
        cache_key = (
            id(clean_dataset), id(patched_dataset), 
            batch_size, total_batches
        )
        
        # Edit the ablation masks to include the always_ablate_fn
        for key, mask in nodes_ablation_mask.items():
            if always_ablate_fn(key):
                # Set the mask to 1 for all positions
                nodes_ablation_mask[key] = torch.ones_like(mask, dtype=torch.bool, device=self.device)

        # If cutoff_early_layers is True, we must not ablate the first layers of the model
        if cutoff_early_layers:
            early_layer_cutoff = self.model_wrapper.n_layers // 3  # First 1/3 of layers
    
            for key in nodes_ablation_mask.keys():
                # Parse something like "blocks.5.hook_resid_post.hook_sae_acts_post"
                layer_str = key.split('.')[1]  # Gets "5" from the example
                layer_num = int(layer_str)
                
                # If node is from early layers, add to restore list
                if layer_num < early_layer_cutoff:
                    nodes_to_restore.append(key)

        # Count how many nodes were part of the circuit
        n_nodes_in_circuit = 0
        for key, mask in nodes_ablation_mask.items():
            n_nodes_in_circuit += torch.sum(~mask).item() # count the number of nodes for which mask is False (i.e. not ablated)

        if correct_for_the_nodes_to_restore:
            # Account for the nodes to restore being not part of the circuit for the calculation of total circuit size
            for key in nodes_to_restore:
                n_nodes_in_circuit -= torch.sum(~nodes_ablation_mask[key]).item()
        
        # Compute the metric values for the circuit C (when nodes outside C are ablated) and the full model M
        # I.e., in SFC paper notation, compute m(C) and m(M) respectively
        if verbose:
            print(f"Running model with circuit nodes ablated ({n_nodes_in_circuit} nodes in circuit) and full model...")
            print(f'Restoring {len(nodes_to_restore)} nodes.')
            if early_layer_cutoff:
                print(f'{early_layer_cutoff * 6} of them are from the first {early_layer_cutoff} layers.') # 6 because there are 3 act types (resid, mlp, attn) and 2 node types (error, sae_post)

        circuit_metrics, full_model_metrics = self._run_model_with_ablation(
            clean_dataset, patched_dataset, 
            batch_size=batch_size, prompts_to_process=prompts_to_process, total_batches=total_batches,
            nodes_to_ablate=nodes_ablation_mask, nodes_to_restore=nodes_to_restore,
            run_full_model=True
        )
        clear_cache()

        # Check if we have cached empty circuit metrics for this configuration
        if cache_key not in self._empty_circuit_metrics_cache:
            # If not in cache, compute empty circuit metrics
            print("Computing empty circuit metrics (will be cached)...")
            
            # Create empty circuit mask (ablate everything)
            empty_circuit_mask = {}
            for key in self.sfc_node_scores.mean_activations.keys():
                empty_circuit_mask[key] = torch.ones_like(nodes_ablation_mask[key], dtype=torch.bool, device=self.device)
            
            # Run evaluation with empty circuit
            empty_circuit_metrics, _ = self._run_model_with_ablation(
                clean_dataset, patched_dataset, 
                batch_size=batch_size, prompts_to_process=prompts_to_process, total_batches=total_batches,
                nodes_to_ablate=empty_circuit_mask, nodes_to_restore=[],
                run_full_model=False
            )
            
            # Cache the result
            self._empty_circuit_metrics_cache[cache_key] = empty_circuit_metrics
        else:
            # Use cached empty circuit metrics
            empty_circuit_metrics = self._empty_circuit_metrics_cache[cache_key]

        # Our metrics are now tensors of shape [total_batches * 2]
        faithfulness_metrics = (circuit_metrics - empty_circuit_metrics) / (full_model_metrics - empty_circuit_metrics)
        
        return faithfulness_metrics, n_nodes_in_circuit
        
    def _run_model_with_ablation(self, clean_dataset: Dict[str, torch.Tensor], patched_dataset: Dict[str, torch.Tensor],
                                 batch_size: int, prompts_to_process: int, total_batches: int,
                                 nodes_to_ablate: Dict[str, torch.Tensor],
                                 nodes_to_restore: List[str] = [],
                                 run_full_model: bool = False,
                                 ablation_values: Optional[Dict[str, torch.Tensor]] = None, 
                                 ) -> Tuple[float, Optional[float]]:
        """
        Run the model with mean ablation on specified nodes at specified positions.
        
        Args:
            clean_dataset: The clean dataset to evaluate on
            patched_dataset: The patched dataset to evaluate on
            nodes_to_ablate: Dictionary mapping node names to binary masks indicating which positions 
                (and features in case of SAE latent nodes) to ablate
            ablation_values: Dictionary mapping node names to their substitution values for ablation.
                Defaults to using mean (position-aware) activation values
            nodes_to_restore: List of node names to restore activation for (i.e. substitute their values as they would be without ablation)
                Defaults to an empty list
            run_full_model: Whether to run the full model without ablation for returning the full model metric value.
                Must be true when nodes_to_restore is not empty.
            batch_size: Number of examples to process at once
            total_batches: Total number of batches to process
            
        Returns:
            (ablated_metric_value, full_model_metric_value),
                if nodes_to_restore is not empty, otherwise just ablated_metric_value.
            This is done to avoid running the full model if not necessary for restoring the activations.

        Note:
            - The returned metric values are Tensors with the shape [total_batches * 2]:
                [
                    metric_value_on_clean_prompts_batch_1,
                    metric_value_on_patched_prompts_batch_1,
                    ...
                    metric_value_on_clean_prompts_batch_n,
                    metric_value_on_patched_prompts_batch_n
                ]
        """
        if nodes_to_restore and not run_full_model:
            print("WARNING: If nodes_to_restore is not empty, run_full_model must be True, but the passed value is False.")
            print("Setting run_full_model to True.")
            run_full_model = True 

        # Perform mean ablation by default
        if ablation_values is None:
            ablation_values = self.sfc_node_scores.mean_activations

        # We'll intervene on every SAE latent node and every error node
        def act_to_hook_on(act_name):
            return 'hook_sae_post' in act_name or 'hook_sae_error' in act_name

        # Also define the hook selection function only for nodes_to_restore
        def act_to_hook_on_restore(act_name):
            return act_name in nodes_to_restore
        
        # Define the hook function for ablation - it will be called for each SAE latent & error node activation during forward pass
        def ablation_hook(act, hook, original_act_cache={}):
            """
            Hook that ablates values for a given activations, for which nodes_to_ablate binary mask is True.
            The substitution values to ablate with are taken from the ablation_values.
            Optionally, if there are any nodes to restore, restores their activations using the original_act_cache
            """
            # First, restore the original activations for the nodes to restore
            if hook.name in nodes_to_restore:
                return original_act_cache[hook.name]

            # Get the mask indicating which nodes to ablate
            ablation_mask = nodes_to_ablate[hook.name]

            # Get the corresponding ablation values
            current_ablation_values = ablation_values[hook.name]

            # ablation mask is of the same shape as the sfc_scores
            #   [pos] for error nodes
            #   [pos, d_sae] for SAE latent nodes

            # but we need to use it to index into current_ablation_values tensors that have the different shape:
            #   [pos, d_model] for error nodes (non-attention)
            #   [pos, n_head, d_head] for attention error nodes
            #   [pos, d_sae] for SAE latent nodes

            # Given that act is of shape
            #   [batch, pos, d_model] for error nodes (non-attention)
            #   [batch, pos, n_head, d_head] for attention error nodes
            #   [batch, pos, d_sae] for SAE latent nodes
            # We do the following re-indexing:

            batch_size = act.shape[0]

            if 'hook_sae_error' in hook.name:
                if 'hook_z' not in hook.name:  # regular errors case
                    # Create a broadcasted mask for the batch and d_model dimensions
                    broadcasted_mask = einops.repeat(ablation_mask, 'pos -> batch pos d_model', 
                                                     batch=batch_size, d_model=act.shape[-1])
                    
                    # Create a selection of ablated values that matches the act shape
                    broadcasted_values = einops.repeat(current_ablation_values, 'pos d_model -> batch pos d_model', 
                                                       batch=batch_size)
                    
                    # Apply the mask to select either original or ablated values
                    act = torch.where(broadcasted_mask, broadcasted_values, act)
                else:  # attention errors case
                    # Create a broadcasted mask for the batch, n_head, and d_head dimensions
                    broadcasted_mask = einops.repeat(ablation_mask, 'pos -> batch pos n_head d_head', 
                                                     batch=batch_size, n_head=act.shape[2], d_head=act.shape[3])
                    
                    # Create a selection of ablated values that matches the act shape
                    broadcasted_values = einops.repeat(current_ablation_values, 'pos n_head d_head -> batch pos n_head d_head', 
                                                       batch=batch_size)
                    
                    # Apply the mask to select either original or ablated values
                    act = torch.where(broadcasted_mask, broadcasted_values, act)
            else:  # SAE latents case
                # For SAE latents, mask has shape [pos, d_sae]
                # Create a broadcasted mask for the batch dimension
                broadcasted_mask = einops.repeat(ablation_mask, 'pos d_sae -> batch pos d_sae', 
                                                 batch=batch_size)
                
                # Create a selection of ablated values that matches the act shape
                broadcasted_values = einops.repeat(current_ablation_values, 'pos d_sae -> batch pos d_sae', 
                                                   batch=batch_size)
                
                # Apply the mask to select either original or ablated values
                act = torch.where(broadcasted_mask, broadcasted_values, act)

            return act
        # end of ablation_hook
        # Now, initialize the metric values
        if run_full_model:
            full_model_metrics_list = []
        ablated_metrics_list = []
        
        # Process each batch
        for i in tqdm(range(0, prompts_to_process, batch_size)):
            # Sample from clean and corrupted datasets
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
                clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            # Start with processing clean prompts
            if run_full_model:
                # Run the model without ablations to collect the original activations (only the ones we want to restore)
                clean_logits, clean_cache = self.model_wrapper.model.run_with_cache(clean_prompts, attention_mask=clean_attn_mask,
                                                                                    names_filter=act_to_hook_on_restore)
                # Calculate logit difference for the clean prompts (if the model is accurate it should be negative, hence the minus in front)
                clean_metric_value = -self.model_wrapper.get_logit_diff(clean_logits, clean_answers=clean_answers,
                                                                        patched_answers=corrupted_answers,
                                                                        answer_pos=clean_answers_pos)
                full_model_metrics_list.append(clean_metric_value.mean().item())

                # Store the original activations for the nodes to restore
                current_hook = lambda act, hook: ablation_hook(act, hook, clean_cache)
            else:
                current_hook = lambda act, hook: ablation_hook(act, hook) # no need to pass original activations
            
            # Run the model with ablations
            ablated_logits = self.model_wrapper.model.run_with_hooks(clean_prompts, attention_mask=clean_attn_mask, 
                                                                     fwd_hooks=[(act_to_hook_on, current_hook)])
            # Calculate logit difference for the clean prompts (if the model is accurate it should be negative, hence the minus in front)
            ablated_metric_value = -self.model_wrapper.get_logit_diff(ablated_logits, clean_answers=clean_answers,
                                                                        patched_answers=corrupted_answers,
                                                                        answer_pos=clean_answers_pos)
            ablated_metrics_list.append(ablated_metric_value.mean().item())

            # Free memory
            del ablated_logits, clean_prompts, clean_answers_pos, clean_attn_mask, current_hook
            if run_full_model:
                del clean_cache, clean_logits
            clear_cache()

            # Now repeat the same for corrupted prompts
            if run_full_model:
                # Run the model without ablations to collect the original activations
                corrupted_logits, corrupted_cache = self.model_wrapper.model.run_with_cache(corrupted_prompts, attention_mask=corrupted_attn_mask,
                                                                                            names_filter=act_to_hook_on_restore)
                # Calculate logit difference for the corrupted prompts (if the model is accurate it should be positive)
                corrupted_metric_value = self.model_wrapper.get_logit_diff(corrupted_logits, clean_answers=clean_answers,
                                                                             patched_answers=corrupted_answers,
                                                                             answer_pos=corrupted_answers_pos)
                full_model_metrics_list.append(corrupted_metric_value.mean().item())

                # Store the original activations for the nodes to restore
                current_hook = lambda act, hook: ablation_hook(act, hook, corrupted_cache)
            else:
                current_hook = lambda act, hook: ablation_hook(act, hook)

            # Run the model with ablations
            ablated_logits = self.model_wrapper.model.run_with_hooks(corrupted_prompts, attention_mask=corrupted_attn_mask, 
                                                                     fwd_hooks=[(act_to_hook_on, current_hook)])
            # Calculate logit difference for the corrupted prompts (if the model is accurate it should be positive)
            ablated_metric_value = self.model_wrapper.get_logit_diff(ablated_logits, clean_answers=clean_answers,
                                                                      patched_answers=corrupted_answers,
                                                                      answer_pos=corrupted_answers_pos)
            ablated_metrics_list.append(ablated_metric_value.mean().item())

            # Free memory
            del ablated_logits, corrupted_prompts, corrupted_answers_pos, corrupted_attn_mask, clean_answers, corrupted_answers, current_hook
            if run_full_model:
                del corrupted_cache, corrupted_logits
            clear_cache()

        # Convert metric lists to tensors and return them
        ablated_metric_tensor = torch.tensor(ablated_metrics_list).to(self.device)
        if run_full_model:
            full_model_metric_tensor = torch.tensor(full_model_metrics_list).to(self.device)
            return ablated_metric_tensor, full_model_metric_tensor
        else:
            return ablated_metric_tensor, None

    def analyze_circuit_composition(self, thresholds, cutoff_early_layers=True):
        """
        Analyze the composition of circuits at different thresholds.
        
        Args:
            thresholds: List of thresholds to analyze
            
        Returns:
            DataFrame with circuit composition analysis
        """
        import pandas as pd
        
        results = []
        
        for threshold in sorted(thresholds, reverse=True):
            masks = self.determine_nodes_to_ablate(threshold)
            
            # Count total nodes and nodes by type
            total_nodes = 0
            feature_nodes = 0
            error_nodes = 0
            resid_error_nodes = 0
            
            for key, mask in masks.items():
                # Count nodes in circuit (where mask is False)
                node_count = torch.sum(~mask).item()
                total_nodes += node_count
                
                if 'hook_sae_error' in key:
                    if 'resid_post' in key:
                        resid_error_nodes += node_count
                    else:
                        error_nodes += node_count
                else:  # SAE feature nodes
                    feature_nodes += node_count
            
            results.append({
                'threshold': threshold,
                'total_nodes': total_nodes,
                'feature_nodes': feature_nodes,
                'non_resid_error_nodes': error_nodes,
                'resid_error_nodes': resid_error_nodes,
                'feature_pct': feature_nodes / total_nodes * 100 if total_nodes > 0 else 0,
                'error_pct': error_nodes / total_nodes * 100 if total_nodes > 0 else 0,
                'resid_error_pct': resid_error_nodes / total_nodes * 100 if total_nodes > 0 else 0
            })
        
        return pd.DataFrame(results)

