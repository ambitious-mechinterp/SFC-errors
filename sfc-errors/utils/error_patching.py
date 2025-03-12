import torch
from tqdm.notebook import tqdm
import gc

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def compute_error_patching_scores(model, clean_dataset, patched_dataset, get_logit_diff_fn, 
                                  batch_size=50, total_batches=None, token_specific_error_types=None, 
                                  token_positions=None, layers_to_patch=None, sample_dataset_fn=None):
    """
    Compute activation patching scores for SAE error terms.
    
    Parameters:
    -----------
    model: TransformerLens model
        The model to compute patching scores for
    clean_dataset: dict
        Dictionary containing the clean prompts, answers, etc.
    patched_dataset: dict
        Dictionary containing the patched/corrupted prompts, answers, etc.
    get_logit_diff_fn: function
        Function that computes logit differences for the model
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
    sample_dataset_fn: function or None
        Function to sample from datasets. If None, uses a default implementation.
    
    Returns:
    --------
    dict
        Dictionary mapping error types to patching scores
    """
    # Set defaults
    if token_specific_error_types is None:
        token_specific_error_types = []  # Empty list means no token-specific patching
    
    # Use provided sample_dataset_fn or the default implementation
    if sample_dataset_fn is None:
        sample_dataset_fn = lambda start_idx, end_idx, clean_dataset, corrupted_dataset: _default_sample_dataset(
            start_idx, end_idx, clean_dataset, corrupted_dataset)
        
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
    n_layers = model.cfg.n_layers
    if layers_to_patch is None:
        layers_to_patch = {
            'resid': list(range(n_layers)),
            'mlp': list(range(n_layers)),
            'attn': list(range(n_layers)),
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
    all_normalized_logit_dif = {
        'resid': torch.zeros((n_layers, seq_len), device=model.cfg.device),
        'mlp': torch.zeros((n_layers, seq_len), device=model.cfg.device),
        'attn': torch.zeros((n_layers, seq_len), device=model.cfg.device)
    }

    # Loop over batches
    for i in range(0, prompts_to_process, batch_size):
        print(f'---\nSamples {i}/{prompts_to_process}\n---')
        clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset_fn(i, i + batch_size, clean_dataset, patched_dataset)

        # Get the corrupted cache (i.e. cache for patched prompts) for all the error nodes
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_prompts, attention_mask=corrupted_attn_mask, 
                                                                 names_filter=lambda name: 'error' in name)
        # Get the clean logits (output of the model on the clean prompts)
        clean_logits = model(clean_prompts, attention_mask=clean_attn_mask)
        
        # Get the logit_diff - difference between incorrect and correct answers' logits - for clean and corrupted prompts
        clean_logit_diff = get_logit_diff_fn(clean_logits, clean_answers=clean_answers,
                                        patched_answers=corrupted_answers,
                                        answer_pos=clean_answers_pos)
        corrupted_logit_diff = get_logit_diff_fn(corrupted_logits, clean_answers=clean_answers,
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
                        logit_diff = get_logit_diff_fn(logits, clean_answers=clean_answers,
                                                       patched_answers=corrupted_answers,
                                                       answer_pos=clean_answers_pos)
                        
                        # Compute normalized logit diff
                        normalized_logit_dif = (logit_diff - clean_logit_diff) / normalized_logit_dif_denom
                        
                        # Store the result for this specific position
                        all_normalized_logit_dif[error_type][layer, pos] += normalized_logit_dif.mean(0)
                        
                        del logits, logit_diff, normalized_logit_dif
                        clear_cache()
            else:
                print("  Using global patching")
                
                # Create a global patching hook
                global_hook = lambda act, hook, cache=corrupted_cache: global_patching_hook(act, hook, cache)
                
                # Loop over layers
                for layer_idx, layer in enumerate(tqdm(layers)):
                    # Apply global patching
                    logits = model.run_with_hooks(clean_prompts, attention_mask=clean_attn_mask, fwd_hooks=[
                        (get_error_activation_name(error_type, layer), global_hook)
                    ])
                    
                    # Compute the logit diff for our patching run
                    logit_diff = get_logit_diff_fn(logits, clean_answers=clean_answers,
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

def _default_sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    """Default implementation for sampling from SFC datasets:
        <> clean_dataset and corrupted_dataset are assumed to be of the same format as returned by SFCDatasetLoader.get_clean_corrupted_datasets()
    
    """
    assert clean_dataset is not None or corrupted_dataset is not None, 'At least one dataset must be provided.'
    return_values = []

    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])

    return return_values