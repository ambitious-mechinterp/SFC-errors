import torch
from pathlib import Path
import json
import pickle
from enum import Enum
from transformer_lens import ActivationCache
from jaxtyping import Float
from torch import Tensor
import einops
import os
from tqdm.notebook import tqdm

class AttributionAggregation(Enum):
    ALL_TOKENS = 'All_tokens'
    NONE = 'None'

class SFC_NodeScores:
    """
    A wrapper around node_scores dictionary with related utilities.
    This class encapsulates the functionality for managing sparse feature circuit scores,
    including initialization, aggregation, saving, and loading.
    """
    def __init__(self, device=None, caching_device=None, control_seq_len=1, 
                 data_dir=None, experiment_name=None, load_if_exists=True):
        """
        Initialize SFC_NodeScores.
        
        Args:
            device: The device to use for computation
            caching_device: The device to use for caching activations
            control_seq_len: The number of tokens to ignore from the beginning of the sequence
            data_dir: Directory to save/load scores
            experiment_name: Name of the experiment for saving/loading
            load_if_exists: Whether to load saved scores if they exist
        """
        self.device = device
        self.caching_device = caching_device if caching_device is not None else device
        self.control_seq_len = control_seq_len
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.experiment_name = experiment_name
        
        # Node scores dictionaries
        self.node_scores = None  # SFC scores - non-aggregated [pos, ...]
        self.aggregated_node_scores = None  # SFC scores - aggregated (sum over positions)
        self.mean_activations = None  # Mean activations - non-aggregated [pos, ...]
        
        # Load saved scores if needed
        if load_if_exists and data_dir is not None and experiment_name is not None:
            self.load_scores()
    
    def initialize_node_scores(self, cache: ActivationCache, run_without_saes=True, 
                               d_sae_lookup_fn=None, hook_name_to_sae_act_name_fn=None):
        """
        Initialize node_scores dictionary based on a cache of activations.
        
        Args:
            cache: Activation cache containing model activations
            run_without_saes: Whether the model is run without SAEs
            d_sae_lookup_fn: Function to get d_sae from a hook name
            hook_name_to_sae_act_name_fn: Function to get SAE act names from a hook name
            
        Returns:
            Initialized node_scores dictionary
        """
        node_scores = {}
        
        for key, cache_tensor in cache.items():
            # A node is either an SAE latent or an SAE error term
            # Here it's represented as the hook-point name - cache key

            if run_without_saes:
                if d_sae_lookup_fn is None or hook_name_to_sae_act_name_fn is None:
                    raise ValueError("d_sae_lookup_fn and hook_name_to_sae_act_name_fn must be provided for no-SAEs runs")
                
                d_sae = d_sae_lookup_fn(key)

                # cache is of shape [batch pos d_act] if not an attn hook, [batch pos n_head d_head] otherwise
                if 'hook_z' not in key:
                    batch, pos, d_act = cache_tensor.shape
                else:
                    batch, pos, n_head, d_act = cache_tensor.shape

                sae_latent_name, sae_error_name = hook_name_to_sae_act_name_fn(key)

                node_scores[sae_error_name] = torch.zeros((pos), dtype=torch.bfloat16, device=self.caching_device)
                node_scores[sae_latent_name] = torch.zeros((pos, d_sae), dtype=torch.bfloat16, device=self.caching_device)
            else:
                if 'hook_z.hook_sae_error' not in key:
                    batch, pos, d_act = cache_tensor.shape
                else:
                    batch, pos, n_head, d_act = cache_tensor.shape

                if 'hook_sae_error' in key:
                    # An "importance value" for the SAE error is scalar - it's a single node
                    node_scores[key] = torch.zeros((pos), dtype=torch.bfloat16, device=self.device)
                else:
                    # An "importance value" for SAE latents is a vector with length d_sae (d_act)
                    node_scores[key] = torch.zeros((pos, d_act), dtype=torch.bfloat16, device=self.device)

        self.node_scores = node_scores
        return node_scores
    
    def initialize_mean_activations(self, cache: ActivationCache, run_without_saes=False):
        """
        Initialize mean_activations dictionary based on a cache of activations.
        
        Args:
            cache: Activation cache containing model activations
            run_without_saes: Whether the model is run without SAEs
        
        Returns:
            Initialized mean_activations dictionary
        """
        mean_activations = {}
        
        for key, cache_tensor in cache.items():
            # For SAEs attached case (run_without_saes=False), initialize with zeros of the same shape
            if 'hook_z.hook_sae_error' not in key:
                # Regular nodes (non-attention)
                batch, pos, d_act = cache_tensor.shape
                mean_activations[key] = torch.zeros((pos, d_act), dtype=torch.bfloat16, device=self.device)
            else:
                # Attention error nodes
                batch, pos, n_head, d_head = cache_tensor.shape
                mean_activations[key] = torch.zeros((pos, n_head, d_head), dtype=torch.bfloat16, device=self.device)
                
        self.mean_activations = mean_activations
        return mean_activations
    
    def aggregate_scores(self, scores_dict, aggregation_type=AttributionAggregation.ALL_TOKENS):
        """
        Aggregate scores across token positions.
        
        Args:
            scores_dict: Dictionary of scores to aggregate
            aggregation_type: Type of aggregation to perform
            
        Returns:
            Dictionary of aggregated scores
        """
        aggregated_scores = {}
        
        for key, score_tensor in scores_dict.items():
            score_tensor_filtered = score_tensor[self.control_seq_len:]
            if aggregation_type == AttributionAggregation.ALL_TOKENS:
                score_tensor_aggregated = score_tensor_filtered.sum(0)
            else:
                score_tensor_aggregated = score_tensor_filtered

            aggregated_scores[key] = score_tensor_aggregated
            
        return aggregated_scores
    
    def aggregate_node_scores(self, aggregation_type=AttributionAggregation.ALL_TOKENS):
        """
        Aggregate node_scores across token positions.
        
        Args:
            aggregation_type: Type of aggregation to perform
        """
        if self.node_scores is None:
            raise ValueError("Node scores have not been initialized")
            
        self.aggregated_node_scores = self.aggregate_scores(self.node_scores, aggregation_type)
        return self.aggregated_node_scores

    def select_node_scores(self, selection_fn, scores_type='sfc'):
        """
        Select node scores based on a selection function.
        
        Args:
            scores_type: Type of scores to select from. Can be 'sfc' or 'mean_act'
            selection_fn: Boolean function called on node names that returns True for selected nodes
        Returns:
            Dictionary of selected node scores

        Notes:
            - The node names are the keys of the node_scores dictionary (e.g. 'blocks.0.hook_resid_post.hook_sae_error).
            - The function doesn't make any copies of the score tensors, so modyfying the returned tensors will modify the original node_scores.
        """
        if scores_type == 'sfc' and self.node_scores is None:
            raise ValueError("Node scores have not been initialized")
        elif scores_type == 'mean_act' and self.mean_activations is None:
            raise ValueError("Mean activations have not been initialized")
        elif scores_type not in ['sfc', 'mean_act']:
            raise ValueError("Invalid scores_type. Must be 'sfc' or 'mean_act'")

        if scores_type == 'sfc':
            scores_dict = self.node_scores
        elif scores_type == 'mean_act':
            scores_dict = self.mean_activations
            
        selected_node_scores = {key: score for key, score in scores_dict.items() if selection_fn(key)}
        return selected_node_scores
    
    def save_scores(self, mode="all"):
        """
        Save scores to data directory.
        
        Args:
            mode: Which scores to save. Can be 'all', 'sfc', or 'mean_act'
        """
        if self.data_dir is None or self.experiment_name is None:
            raise ValueError("data_dir and experiment_name must be set to save scores")
            
        # Create directory if it doesn't exist
        save_dir = self.data_dir / self.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file with information about the scores
        metadata = {
            "control_seq_len": self.control_seq_len,
            "has_sfc_scores": self.node_scores is not None,
            "has_aggregated_sfc_scores": self.aggregated_node_scores is not None, 
            "has_mean_activations": self.mean_activations is not None,
        }
        
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save scores
        if (mode == "all" or mode == "sfc") and self.node_scores is not None:
            print(f"Saving SFC scores to {save_dir}")
            self._save_tensor_dict(self.node_scores, save_dir / "sfc_scores.pkl")
            
        if (mode == "all" or mode == "sfc") and self.aggregated_node_scores is not None:
            print(f"Saving aggregated SFC scores to {save_dir}")
            self._save_tensor_dict(self.aggregated_node_scores, save_dir / "aggregated_sfc_scores.pkl")
            
        if (mode == "all" or mode == "mean_act") and self.mean_activations is not None:
            print(f"Saving mean activations to {save_dir}")
            self._save_tensor_dict(self.mean_activations, save_dir / "mean_activations.pkl")
    
    def load_scores(self, map_location=None):
        """
        Load scores from data directory.
        
        Args:
            map_location: Optional argument to specify a location for mapping tensors
                          when loading. Can be a device, string, dict, or a function.
                          Default: None (load to same device as when saved)
        """
        if self.data_dir is None or self.experiment_name is None:
            raise ValueError("data_dir and experiment_name must be set to load scores")
            
        load_dir = self.data_dir / self.experiment_name
        
        # Check if directory exists
        if not load_dir.exists():
            print(f"No saved scores found at {load_dir}")
            return
            
        # Load metadata if it exists
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Set control_seq_len from metadata
            if "control_seq_len" in metadata:
                self.control_seq_len = metadata["control_seq_len"]
        
        # Define all possible file paths (including .pt for backward compatibility)
        file_paths = {
            "sfc_scores": [load_dir / "sfc_scores.pkl", load_dir / "sfc_scores.pt"],
            "aggregated_sfc_scores": [load_dir / "aggregated_sfc_scores.pkl", load_dir / "aggregated_sfc_scores.pt"],
            "mean_activations": [load_dir / "mean_activations.pkl", load_dir / "mean_activations.pt"]
        }
        
        # Try loading from each possible path
        for score_type, paths in file_paths.items():
            for path in paths:
                if path.exists():
                    if score_type == "sfc_scores":
                        self.node_scores = self._load_tensor_dict(path, map_location)
                        print(f"Loaded SFC scores from {path}")
                    elif score_type == "aggregated_sfc_scores":
                        self.aggregated_node_scores = self._load_tensor_dict(path, map_location)
                        print(f"Loaded aggregated SFC scores from {path}")
                    elif score_type == "mean_activations":
                        self.mean_activations = self._load_tensor_dict(path, map_location)
                        print(f"Loaded mean activations from {path}")
                    break  # Once loaded, move to next score type
    
    def _save_tensor_dict(self, tensor_dict, path):
        """
        Helper method to save a dictionary of tensors.
        Saves tensors directly in their current device without moving to CPU.
        
        Args:
            tensor_dict: Dictionary of tensors to save
            path: Path where to save the dictionary
        """
        # Save directly without detaching or moving to CPU
        torch.save(tensor_dict, path)
    
    def _load_tensor_dict(self, path, map_location=None):
        """
        Helper method to load a dictionary of tensors.
        
        Args:
            path: Path to the saved tensor dictionary
            map_location: Optional parameter to specify a location to map tensors to
                          when loading. Can be a device, string, dict, or a function.
                          Default: None (load to same device as when saved)
                          
        Returns:
            Dictionary of loaded tensors
        """
        if map_location is None and self.device is not None:
            # Use the instance's device by default
            map_location = self.device
            
        tensor_dict = torch.load(path, map_location=map_location)
        return tensor_dict