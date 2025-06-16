import torch
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
import gc
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
from jaxtyping import Float, Int
from torch import Tensor
import einops
import inspect

# Import SFC_NodeScores
from .sfc_node_scores import SFC_NodeScores
from .sfc_model import sample_dataset

# Utility to clear variables out of the memory & clearing cuda cache
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

class CircuitEvaluator:
    """
    Class for evaluating circuit faithfulness, 
    where circuit is generally understood as any subset of SAE/error nodes that are not ablated.
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
    
    def evaluate_circuit_faithfulness(self, clean_dataset, patched_dataset, node_threshold=None, 
                                      nodes_to_always_ablate=None, nodes_to_restore=None,
                                      always_ablate_positions=None,
                                      feature_indices_to_ablate: Optional[Dict[str, List[int]]] = None,
                                      feature_indices_to_restore=None,
                                      cutoff_early_layers=True,
                                      correct_for_the_nodes_to_restore=False,
                                      use_zero_ablation: bool = False,
                                      batch_size=100, total_batches=None,
                                      verbose=True,
                                      return_all_metrics=False,
                                      _return_components_for_verification: bool = False
                                      ) -> Tuple[torch.Tensor, int]:
        """
        Evaluate the faithfulness of a circuit by selectively ablating and restoring nodes.
        
        Args:
            clean_dataset: The clean dataset to evaluate on. 
                It's the main dataset if only using a single dataset with true/false answers.
            patched_dataset: The patched dataset to evaluate on. 
                Can be None if using only a single dataset with true/false answers.
            node_threshold: Threshold to determine circuit nodes. 
                            If None, operates entirely based on nodes_to_always_ablate and nodes_to_restore

            nodes_to_always_ablate: a List containing the names of the nodes to unconditionally ablate,
                                    or a Callable that maps node names to True/False indicating which nodes to ablate
            always_ablate_positions: an optional List/array/tensor of positions at which we ablate the nodes from `nodes_to_always_ablate`
                                     and/or `feature_indices_to_ablate`. If not given, all positions are ablated by default.
            feature_indices_to_ablate: Dictionary mapping SAE feature node names to lists of feature indices to ablate.
                                       Only applies to SAE feature nodes (hook_sae_acts_post).
            use_zero_ablation: Whether to use zero ablation instead of mean ablation

            nodes_to_restore: List of node names to restore activation for
            feature_indices_to_restore: Dictionary mapping node names to lists of feature indices to restore.
                                        Only applies to SAE feature nodes (hook_sae_acts_post).
                                        If provided for a node, only the specified indices will be restored.

            cutoff_early_layers: Whether to exclude early layers (first 1/3 of the model) from ablation
            correct_for_the_nodes_to_restore: Whether to correct the number of nodes in the circuit for the nodes to restore
            return_all_metrics: Whether to return a tensor of all 2*N faithfulness scores for each of N prompts from the clean and patched datasets
                                Default is False, which means that the scores are aggregated across `total_batches` batches

            batch_size: Number of examples to process at once
            total_batches: Total number of batches to process
            verbose: Whether to print progress information
            _return_components_for_verification: Whether to return the individual components of the faithfulness metric for verification purposes.
        Returns:
            - Faithfulness metrics tensor of shape [total_batches * 2] (2 multiplier comes from the use of both clean & patched dataset)
              OR of shape [total_samples * 2] if return_all_metrics = True,
            - Total number of nodes in the circuit
        """
        # Check if we have the mean activations pre-computed
        if self.sfc_node_scores.mean_activations is None:
            raise ValueError("Mean activations not computed. Compute mean activations first.")
        
        if self.sfc_node_scores.node_scores is None:
            raise ValueError("SFC scores not computed. Compute SFC scores first.")

        if set(self.sfc_node_scores.mean_activations.keys()) != set(self.sfc_node_scores.node_scores.keys()):
            raise ValueError("Mean activations and SFC scores keys do not match.")

        # Count total batches and prompts to process
        n_prompts, seq_len = clean_dataset['prompt'].shape
        if patched_dataset is not None:
            assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]
        else:
            # Handle single dataset case
            true_answer_key = 'true_answer' if 'true_answer' in clean_dataset else 'answer'
            if true_answer_key not in clean_dataset:
                raise KeyError(f"clean_dataset must have 'true_answer' or 'answer' key.")
            if 'false_answer' not in clean_dataset:
                raise KeyError(f"clean_dataset must have 'false_answer' key.")
            
            assert n_prompts == clean_dataset[true_answer_key].shape[0]
            assert n_prompts == clean_dataset['false_answer'].shape[0]

            if verbose:
                print(f"Using single dataset for evaluation.")

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size
            if n_prompts % batch_size != 0:
                total_batches += 1
        
        # Initialize the ablation masks dictionary, mapping node names to their ablation masks
        ablation_masks = {}     

        # Step 1: populate it with full masks for the nodes_to_always_ablate:
        if nodes_to_always_ablate is not None:
            if inspect.isfunction(nodes_to_always_ablate):
                always_ablate_condition = nodes_to_always_ablate
            elif isinstance(nodes_to_always_ablate, list):
                always_ablate_condition = lambda node_name: node_name in nodes_to_always_ablate
            else:
                raise ValueError("nodes_to_always_ablate argument should be either a function or a list of node names")
            
            # Populate binary masks for nodes in nodes_to_always_ablate with True values, indicating ablation at all positions
            for node_name, sfc_scores in self.sfc_node_scores.node_scores.items():
                if always_ablate_condition(node_name):
                    ablation_masks[node_name] = torch.ones_like(sfc_scores, dtype=torch.bool, device=self.device)

                    if always_ablate_positions is not None:
                        # Use only specific ablation position if provided
                        ablation_masks[node_name][always_ablate_positions, ...] = False
                        # Flip the mask because False signifies the positions to NOT ablate (and we want vice-versa)
                        ablation_masks[node_name] = ~ablation_masks[node_name]

        # Handle feature_indices_to_ablate (specific feature ablation)
        if feature_indices_to_ablate is not None:
            # validate that it only contains SAE feature nodes
            for node_name, indices in feature_indices_to_ablate.items():
                if "hook_sae_acts_post" not in node_name:
                    raise ValueError(f"Feature indices can only be ablated for SAE feature nodes, but got {node_name}")
                
                # If a mask for this node doesn't exist, create it as all-False
                if node_name not in ablation_masks:
                    ablation_masks[node_name] = torch.zeros_like(self.sfc_node_scores.node_scores[node_name], dtype=torch.bool, device=self.device)

                indices_tensor = torch.LongTensor(indices).to(self.device)
                if always_ablate_positions is not None:
                    # Ablate specific features at specific positions
                    pos_tensor = torch.LongTensor(always_ablate_positions).to(self.device)
                    # Use broadcasting to select the cross-product of positions and indices
                    ablation_masks[node_name][pos_tensor[:, None], indices_tensor] = True
                else:
                    # Ablate specific features at all positions
                    ablation_masks[node_name][:, indices_tensor] = True
        
        # If node_threshold is provided, use it to determine additional nodes to ablate
        if node_threshold is not None:
            threshold_ablation_masks = self.determine_nodes_to_ablate(node_threshold)
            
            # Merge with existing (always-ablate) masks using logical OR
            for node_name, mask in threshold_ablation_masks.items():
                if node_name in ablation_masks:
                    ablation_masks[node_name] = ablation_masks[node_name] | mask
                else:
                    ablation_masks[node_name] = mask
        else:
            # If no threshold provided, ensure we have masks for all nodes by creating masks that ablate nothing (all False)
            for node_name, sfc_scores in self.sfc_node_scores.node_scores.items():
                if node_name not in ablation_masks:
                    ablation_masks[node_name] = torch.zeros_like(sfc_scores, dtype=torch.bool, device=self.device)

        # At this point we should have a complete set of masks in ablation_masks
        assert set(ablation_masks.keys()) == set(self.sfc_node_scores.node_scores.keys())

        # Step 2: Handle the nodes that we want to restore activations for (i.e. patch in their activations as they would have been without ablation)
        if nodes_to_restore is None:
            nodes_to_restore = []
        
        # Here there are two options: full restoration and restoration only of specific features
        if feature_indices_to_restore is None:
            feature_indices_to_restore = {}
        else:
            # Validate that feature_indices_to_restore only contains SAE feature nodes
            for node_name in feature_indices_to_restore:
                if "hook_sae_acts_post" not in node_name:
                    raise ValueError(f"Feature indices can only be restored for SAE feature nodes, but got {node_name}")
                if node_name not in nodes_to_restore:
                    # Automatically add to nodes_to_restore if not already there
                    nodes_to_restore.append(node_name)
        
        # If cutoff_early_layers is True, we don't ablate the first layers of the model
        if cutoff_early_layers:
            early_layer_cutoff = self.model_wrapper.n_layers // 3  # First 1/3 of layers

            for key in ablation_masks.keys():
                # Parse act name like this "blocks.5.hook_resid_post.hook_sae_acts_post"
                layer_str = key.split('.')[1]  # Gets "5" from the example
                layer_num = int(layer_str)
                
                # If node is from early layers, add to restore list and clear its ablation mask
                if layer_num < early_layer_cutoff:
                    if key not in nodes_to_restore:
                        nodes_to_restore.append(key)

        # Step 3: count how many nodes will be in the circuit (nodes that are not being ablated)
        n_nodes_in_circuit = 0
        for key, mask in ablation_masks.items():
            n_nodes_in_circuit += torch.sum(~mask).item()  # count the number of nodes for which mask is False (i.e. not ablated)

        if correct_for_the_nodes_to_restore:
            # Account for the nodes to restore being not part of the circuit for the calculation of total circuit size
            for key in nodes_to_restore:
                if key in feature_indices_to_restore:
                    # Only count the specific feature indices being restored
                    feature_count = len(feature_indices_to_restore[key])
                    n_nodes_in_circuit -= feature_count
                else:
                    # For full node restoration, count all unablated positions
                    n_nodes_in_circuit -= torch.sum(~ablation_masks[key]).item()
        
        # Optimization: Initialize empty circuit metrics cache if it doesn't exist
        if not hasattr(self, '_empty_circuit_metrics_cache'):
            self._empty_circuit_metrics_cache = {}
        
        # Create a cache key based on dataset properties and other parameters that might affect the metric
        # This is needed so that we don't need to recompute the empty circuit metric for the same dataset and parameters
        cache_key = (
            id(clean_dataset), id(patched_dataset), 
            batch_size, total_batches
        )
        # Optional logging
        if verbose:
            if node_threshold is not None:
                print(f"Running model with nodes outside the circuit ablated ({n_nodes_in_circuit} nodes in circuit) and full model...")
                
            print(f'Restoring {len(nodes_to_restore)} nodes.')
            feature_restore_count = sum(len(indices) for indices in feature_indices_to_restore.values())
            
            if feature_restore_count > 0:
                print(f'Restoring {feature_restore_count} specific features across {len(feature_indices_to_restore)} nodes.')
            if cutoff_early_layers:
                print(f'Not ablating first {early_layer_cutoff} layers, resulting in restorations of {early_layer_cutoff * 6} activation nodes.')

        # Step 4: compute the metric values for the circuit C (when nodes outside C are ablated) and the full model M
        # I.e., in SFC paper notation, compute m(C) and m(M) respectively
        circuit_metrics, full_model_metrics = self._run_model_with_ablation(
            clean_dataset, patched_dataset, 
            batch_size=batch_size, prompts_to_process=prompts_to_process, total_batches=total_batches,
            nodes_to_ablate=ablation_masks, nodes_to_restore=nodes_to_restore,
            feature_indices_to_restore=feature_indices_to_restore,
            use_zero_ablation=use_zero_ablation,
            run_full_model=True,
            verbose=verbose,
            return_all_metrics=return_all_metrics
        )
        clear_cache()

        # Step 5: Compute an empty circuit metric m(∅)
        # First, check if we have cached empty circuit metrics for this configuration
        if cache_key not in self._empty_circuit_metrics_cache:
            # If not in cache, compute empty circuit metrics
            print("Computing empty circuit metrics (will be cached)...")
            
            # Create empty circuit mask (ablate everything)
            empty_circuit_mask = {}
            for key, scores in self.sfc_node_scores.node_scores.items():
                empty_circuit_mask[key] = torch.ones_like(scores, dtype=torch.bool, device=self.device)
            
            # Run evaluation with empty circuit
            empty_circuit_metrics, _ = self._run_model_with_ablation(
                clean_dataset, patched_dataset, 
                batch_size=batch_size, prompts_to_process=prompts_to_process, total_batches=total_batches,
                nodes_to_ablate=empty_circuit_mask, nodes_to_restore=[],
                feature_indices_to_restore={},
                run_full_model=False,
                verbose=verbose,
                return_all_metrics=return_all_metrics
            )
            
            # Cache the result
            self._empty_circuit_metrics_cache[cache_key] = empty_circuit_metrics
        else:
            print("Using cached empty circuit metrics.")
            # Use cached empty circuit metrics
            empty_circuit_metrics = self._empty_circuit_metrics_cache[cache_key]

        faithfulness_metrics = (circuit_metrics - empty_circuit_metrics) / (full_model_metrics - empty_circuit_metrics)
        if _return_components_for_verification:
            return faithfulness_metrics, n_nodes_in_circuit, circuit_metrics, full_model_metrics, empty_circuit_metrics
        else:
            return faithfulness_metrics, n_nodes_in_circuit

    def _run_model_with_ablation(self, clean_dataset: Dict[str, torch.Tensor], patched_dataset: Dict[str, torch.Tensor],
                                 batch_size: int, prompts_to_process: int, total_batches: int,
                                 nodes_to_ablate: Dict[str, torch.Tensor],
                                 nodes_to_restore: List[str] = [],
                                 feature_indices_to_restore: Dict[str, List[int]] = None,
                                 run_full_model: bool = False,
                                 ablation_values: Optional[Dict[str, torch.Tensor]] = None, 
                                 verbose: bool = False,
                                 return_all_metrics=False,
                                 use_zero_ablation: bool = False,
                                 ) -> Tuple[float, Optional[float]]:
        """
        Run the model with mean ablation on specified nodes at specified positions.
        Optionally, restore the activations of other specified nodes, i.e. patch in their activations as they would have been without ablation.
        
        Args:
            clean_dataset: The clean dataset to evaluate on. 
                It's the main dataset if only using a single dataset with true/false answers.
            patched_dataset: The patched dataset to evaluate on. 
                Can be None if using only a single dataset with true/false answers.
                In this case the logit difference is computed between true_answer and false_answer tokens.

            nodes_to_ablate: Dictionary mapping node names to binary masks indicating which positions 
                             (and features in case of SAE latent nodes) to ablate

            nodes_to_restore: List of node names to restore activation for (i.e. substitute their values as they would be without ablation)
                              Defaults to an empty list
            feature_indices_to_restore: Dictionary mapping node names to lists of feature indices to restore.
                                        Only applies to SAE feature nodes (hook_sae_acts_post).
                                        If provided for a node, only the specified indices will be restored.

            ablation_values: Dictionary mapping node names to their substitution values for ablation.
                             Defaults to using mean (position-aware) activation values
            use_zero_ablation: Whether to use zero ablation instead of mean ablation
            run_full_model: Whether to run the full model without ablation for returning the full model metric value.
                            Must be true when nodes_to_restore is not empty, because we need the original cache to restore the activations.

            batch_size: Number of examples to process at once
            total_batches: Total number of batches to process
            return_all_metrics: Whether to return a tensor of all 2*N faithfulness scores for each of N prompts from the clean and patched datasets
                                Default is False, which means that the scores are aggregated across `total_batches` batches
            
        Returns:
            Tuple containing:
            - ablated_metric_tensor: Metrics with ablations applied
            - full_model_metric_tensor: Metrics from full model (if run_full_model=True)
            where metric is a logit dif between correct and incorrect answer tokens
        """
        if nodes_to_restore and not run_full_model:
            print("WARNING: If nodes_to_restore is not empty, run_full_model must be True, but the passed value is False.")
            print("Setting run_full_model to True.")
            run_full_model = True 

        # Initialize feature_indices_to_restore if None
        if feature_indices_to_restore is None:
            feature_indices_to_restore = {}

        if ablation_values is None:
            if use_zero_ablation:
                # Create a dictionary of zero tensors with the correct shape, dtype, and device
                ablation_values = {}
                for key, mean_act_tensor in self.sfc_node_scores.mean_activations.items():
                    ablation_values[key] = torch.zeros_like(mean_act_tensor)
                if verbose:
                    print("Using ZERO ABLATION.")
            else:
                # Perform mean ablation by default
                ablation_values = self.sfc_node_scores.mean_activations
                if verbose:
                    print("Using MEAN ABLATION.")
        else:
            if verbose:
                print("Using custom provided ablation_values.")

        # We'll intervene on every SAE latent node and every error node
        def act_to_hook_on(act_name):
            return 'hook_sae_acts_post' in act_name or 'hook_sae_error' in act_name

        # Also define the hook selection function only for nodes_to_restore
        def act_to_hook_on_restore(act_name):
            return act_name in nodes_to_restore
        
        # Define the hook function for ablation - it will be called for each SAE latent & error node activation during forward pass
        def ablation_hook(act, hook, original_act_cache=None):
            """
            Hook that ablates values for a given activations, for which nodes_to_ablate binary mask is True.
            The substitution values to ablate with are taken from the ablation_values.
            Optionally, if there are any nodes to restore, restores their activations using the original_act_cache
            """
            if original_act_cache is None:
                original_act_cache = {} # Create a new dict for each call

            # First, restore the original activations for the nodes to restore
            if hook.name in nodes_to_restore:
                if verbose:
                    print(f'Restoring {hook.name}')
                # SAE feature nodes might have specific indices to restore
                if hook.name in feature_indices_to_restore and "hook_sae_acts_post" in hook.name:
                    # Get the indices to restore
                    indices = feature_indices_to_restore[hook.name]
                    if verbose:
                        print(f'    using specific indicies: {indices}')
                    
                    # Get the original activations
                    original_acts = original_act_cache[hook.name]
                    
                    # Restore only the specified indices
                    # act has shape [batch, pos, d_sae]
                    # We need to restore act[:, :, indices] from original_acts[:, :, indices]
                    act[:, :, indices] = original_acts[:, :, indices]
                    
                    return act
                else:
                    # For other nodes, restore the entire activation
                    return original_act_cache[hook.name]

            # Get the mask indicating which nodes to ablate
            ablation_mask = nodes_to_ablate[hook.name]
            if verbose and ablation_mask.sum().item() > 0:
                print(f'Ablating {hook.name} in {ablation_mask.sum().item()} positions')
                if "hook_sae_acts_post" in hook.name:
                    print(f'    with {ablation_mask.sum(dim=-1).tolist()} features')

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
                if not return_all_metrics:
                    full_model_metrics_list.append(clean_metric_value.mean().item())
                else:
                    full_model_metrics_list.append(clean_metric_value)

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
            if not return_all_metrics:
                ablated_metrics_list.append(ablated_metric_value.mean().item())
            else:
                ablated_metrics_list.append(ablated_metric_value)

            # Free memory
            del ablated_logits, clean_prompts, clean_answers_pos, clean_attn_mask, current_hook
            if run_full_model:
                del clean_cache, clean_logits
            clear_cache()

            # Now repeat the same for corrupted prompts
            verbose = False # no need to log anymore

            if corrupted_prompts is not None:
                if run_full_model:
                    # Run the model without ablations to collect the original activations
                    corrupted_logits, corrupted_cache = self.model_wrapper.model.run_with_cache(corrupted_prompts, attention_mask=corrupted_attn_mask,
                                                                                                names_filter=act_to_hook_on_restore)
                    # Calculate logit difference for the corrupted prompts (if the model is accurate it should be positive)
                    corrupted_metric_value = self.model_wrapper.get_logit_diff(corrupted_logits, clean_answers=clean_answers,
                                                                                patched_answers=corrupted_answers,
                                                                                answer_pos=corrupted_answers_pos)

                    if not return_all_metrics:                                                         
                        full_model_metrics_list.append(corrupted_metric_value.mean().item())
                    else:
                        full_model_metrics_list.append(corrupted_metric_value)

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
                if not return_all_metrics:
                    ablated_metrics_list.append(ablated_metric_value.mean().item())
                else:
                    ablated_metrics_list.append(ablated_metric_value)

                # Free memory
                del ablated_logits, corrupted_prompts, corrupted_answers_pos, corrupted_attn_mask, clean_answers, corrupted_answers, current_hook
                if run_full_model:
                    del corrupted_cache, corrupted_logits
                clear_cache()

        # Convert metric lists to tensors and return them
        if not return_all_metrics:
            ablated_metric_tensor = torch.tensor(ablated_metrics_list).to(self.device)
        else:
            ablated_metric_tensor = torch.concat(ablated_metrics_list).to(self.device)
        
        if run_full_model:
            if not return_all_metrics: 
                full_model_metric_tensor = torch.tensor(full_model_metrics_list).to(self.device)
            else:
                full_model_metric_tensor = torch.concat(full_model_metrics_list).to(self.device)

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

