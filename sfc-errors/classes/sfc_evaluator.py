import torch
from tqdm.notebook import tqdm
import gc
from enum import Enum

# Import SFC_NodeScores
from .sfc_node_scores import SFC_NodeScores

# Utility to clear variables out of the memory & clearing cuda cache
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

class CircuitType(Enum):
    FEATURES = "features"
    FEATURES_WO_ERRORS = "features_wo_errs"
    FEATURES_WO_SOME_ERRORS = "features_wo_some_errs"
    NEURONS = "neurons"

class CircuitEvaluator:
    """
    Class for evaluating circuit faithfulness and completeness.
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
                        # We treat both feature and error activations the same way
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