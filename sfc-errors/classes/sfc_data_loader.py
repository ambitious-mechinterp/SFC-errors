from datasets import load_dataset, Dataset
from utils.enums import SpecialTokens, SupportedDatasets, DatasetCategory
import torch
import random
from tqdm import tqdm
import os
from pathlib import Path

### Utility functions ###
def find_first_index(tensor: torch.Tensor, value) -> int:
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")

    # Create a boolean mask where the condition is met
    mask = tensor == value

    # Use nonzero to find the indices of True values and get the first occurrence
    indices = mask.nonzero(as_tuple=True)[0]
    
    if indices.numel() > 0:
        return indices[0].item()
    else:
        raise ValueError(f"{value} is not in the tensor")

class SFCDatasetLoader:
    def __init__(self, dataset_name, model, task_prompt='', clean_system_prompt='', corrupted_system_prompt='', split="train", 
                 num_samples=None, local_dataset=False, base_folder_path='./data'):
        self.dataset = self.load_supported_dataset(dataset_name, split, local_dataset, base_folder_path)
        self.dataset_name = dataset_name
        self.base_folder_path = base_folder_path

        self.task_prompt = task_prompt
        self.clean_system_prompt = clean_system_prompt
        self.corrupted_system_prompt = corrupted_system_prompt

        if not clean_system_prompt:
            print('WARNING: Clean system prompt not provided.')
        if not corrupted_system_prompt:
            print('WARNING: Corrupted system prompt not provided.')
        if not task_prompt:
            print('WARNING: Task prompt not provided.')

        self.model = model
        self.device = model.cfg.device
        self.default_special_tokens_tensor = self._get_special_tokens_tensor()

        # Sample the dataset if num_samples is specified
        if num_samples is not None and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(random.sample(range(len(self.dataset)), num_samples))

    def save_subset_by_indices(self, indices: list, filename_suffix: str):
        """
        Selects a subset of the original dataset by indices and saves it to a new JSON file.

        Args:
            indices (list): A list of integer indices to select from the original dataset.
            filename_suffix (str): A descriptive suffix to append to the output filename.
                                   (e.g., "_mm_gt_1" will result in a file like '..._mm_gt_1.json')
        """
        if not isinstance(indices, list):
            indices = list(indices)
            
        print(f"Selecting {len(indices)} samples from the original dataset...")
        subset_dataset = self.dataset.select(indices)

        # Construct the output file path
        file_name_prefix = self.dataset_name.value.replace('/', '_')
        file_name = f"{file_name_prefix}{filename_suffix}.json"
        
        # Use pathlib for robust path creation
        output_path = Path(self.base_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / file_name

        # Save the subset to the new JSON file
        print(f"Saving filtered dataset with {len(subset_dataset)} samples to: {full_path}")
        subset_dataset.to_json(full_path)
        print("Save complete.")

    def filter_and_set_max_length(self, apply_chat_template=True, prepend_generation_prefix=False, filter_long_sequences=True, use_most_common_length=False):
        """
        Calculate optimal padding length and optionally filter outlier-length prompts.
        If all prompts have the same length, no filtering is applied.
        
        Args:
            apply_chat_template (bool): Whether to apply chat template when calculating lengths
            prepend_generation_prefix (bool): Whether to prepend generation prefix
            filter_long_sequences (bool): Whether to filter out sequences longer than 99th percentile
            use_most_common_length (bool): Whether to filter to only the most common prompt length
        """
        # Check if the dataset category requires padding
        if not self.dataset_name.category.requires_padding:
            print(f"Dataset category {self.dataset_name.category.name} doesn't require padding. Skipping max length calculation.")
            self._max_prompt_length = None
            return None
        
        def get_tokenized_length(prompt):
            tokenizer = self.model.tokenizer

            # Apply chat template if required
            if apply_chat_template:
                conversation = [
                    {"role": "user", "content": prompt}
                ]
                prompt = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    continue_final_message = False if prepend_generation_prefix else True,
                    add_generation_prompt = prepend_generation_prefix
                )
            
            if apply_chat_template:
                # Tokenize using the tokenizer with padding and truncation
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=False,
                    return_special_tokens_mask=False
                )
                prompt = tokenized['input_ids'].squeeze(0)
            else:
                # Tokenize using the tokenizer with padding and truncation, and add special tokens
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=True,
                    return_special_tokens_mask=False
                )
                    
                prompt = tokenized["input_ids"].squeeze(0)  # Padded input ID

            return prompt.size(0)
        
        clean_prompts = [self.get_formatted_prompt(item, system_prompt=self.clean_system_prompt, 
                                             task_prompt=self.task_prompt) 
                                             for item in self.dataset]
        corrupted_prompts = [self.get_formatted_prompt(item, system_prompt=self.corrupted_system_prompt, 
                                             task_prompt=self.task_prompt) 
                                             for item in self.dataset]
        
        clean_prompts_lengths = torch.tensor([get_tokenized_length(prompt) for prompt in clean_prompts])
        corrupted_prompts_lengths = torch.tensor([get_tokenized_length(prompt) for prompt in corrupted_prompts])
        prompts_count = clean_prompts_lengths.size(0)

        # Check for unique lengths
        unique_clean_lengths = torch.unique(clean_prompts_lengths)
        unique_corrupted_lengths = torch.unique(corrupted_prompts_lengths)
        
        # Check if all prompts have the same length
        if len(unique_clean_lengths) == 1 and len(unique_corrupted_lengths) == 1:
            # All clean prompts have one length, all corrupted prompts have one length
            clean_length = unique_clean_lengths.item()
            corrupted_length = unique_corrupted_lengths.item()
            print(f"All prompts have uniform lengths. Clean: {clean_length}, Corrupted: {corrupted_length}")
            self._max_prompt_length = max(clean_length, corrupted_length)
            print(f'Setting max prompt length to {self._max_prompt_length}')
            return self._max_prompt_length
        
        # If we get here, there is actual variation in prompt lengths
        clean_min = clean_prompts_lengths.min().item()
        clean_max = clean_prompts_lengths.max().item()
        corrupted_min = corrupted_prompts_lengths.min().item()
        corrupted_max = corrupted_prompts_lengths.max().item()
        
        print(f"Found variation in prompt lengths. Clean: {clean_min}-{clean_max} ({len(unique_clean_lengths)} unique lengths)")
        print(f"Corrupted: {corrupted_min}-{corrupted_max} ({len(unique_corrupted_lengths)} unique lengths)")
        
        # Check if the variation is minimal (e.g., only a few outliers)
        # Calculate how many prompts have the most common length
        clean_mode = torch.mode(clean_prompts_lengths).values.item()
        clean_mode_count = (clean_prompts_lengths == clean_mode).sum().item()
        clean_pct_mode = (clean_mode_count / len(clean_prompts_lengths)) * 100
        
        corrupted_mode = torch.mode(corrupted_prompts_lengths).values.item()
        corrupted_mode_count = (corrupted_prompts_lengths == corrupted_mode).sum().item()
        corrupted_pct_mode = (corrupted_mode_count / len(corrupted_prompts_lengths)) * 100
        
        print(f"Clean: Most common length is {clean_mode} ({clean_pct_mode:.1f}% of prompts)")
        print(f"Corrupted: Most common length is {corrupted_mode} ({corrupted_pct_mode:.1f}% of prompts)")
        
        if use_most_common_length:
            filtered_indices = [i for i in range(prompts_count) if clean_prompts_lengths[i] == clean_mode and corrupted_prompts_lengths[i] == corrupted_mode]
            
            if not filtered_indices:
                print("Warning: No prompts found with the most common length for both clean and corrupted versions. No filtering applied.")
                length_threshold = max(clean_prompts_lengths.max().item(), corrupted_prompts_lengths.max().item())
            else:
                self.dataset = self.dataset.select(filtered_indices)
                num_filtered = prompts_count - len(filtered_indices)
                print(f"Filtered dataset to most common prompt lengths. Removed {num_filtered} of {prompts_count} prompts.")
                length_threshold = max(clean_mode, corrupted_mode)

        elif filter_long_sequences:
            # Step 2: Calculate the length threshold for the top 1% longest entries
            clean_threshold_length = torch.quantile(clean_prompts_lengths.float(), 0.99).item()
            corrupted_threshold_length = torch.quantile(corrupted_prompts_lengths.float(), 0.99).item()
            length_threshold = max(clean_threshold_length, corrupted_threshold_length)
        
            # Step 3: Filter out entries where length is greater than or equal to the threshold
            filtered_indices_clean = [i for i, length in enumerate(clean_prompts_lengths) if length < clean_threshold_length]
            filtered_indices_corrupted = [i for i, length in enumerate(corrupted_prompts_lengths) if length < corrupted_threshold_length]
            filtered_indices = list(set(filtered_indices_clean).intersection(filtered_indices_corrupted))

            filtered_dataset = self.dataset.select(filtered_indices)
            self.dataset = filtered_dataset

            # Print the number of filtered elements
            num_filtered = prompts_count - len(filtered_indices)
            print(f"Filtered out {num_filtered} longest prompts from a total of {prompts_count} prompts.")
        else:
            print("Skipping sequence length filtering. Using max prompt length for padding.")
            length_threshold = max(clean_prompts_lengths.max().item(), corrupted_prompts_lengths.max().item())

        self._max_prompt_length = int(length_threshold)
        print(f'Setting max prompt length to {self._max_prompt_length}')

        return self._max_prompt_length

    def _get_special_tokens_tensor(self, selected_special_tokens=None):
        """
        Returns a tensor of selected special tokens from the tokenizer and custom tokens ['user', 'model'].
        If `selected_special_tokens` is None, all tokens from the given set are included.
        
        Parameters:
        - selected_special_tokens (list[SpecialTokens], optional): A list of special tokens to include. 
                                                                If None, all are included by default.
        
        Returns:
        - torch.Tensor: Tensor of selected special tokens' IDs.
        """

        # Step 2: Set default tokens if none are provided
        if selected_special_tokens is None:
            selected_special_tokens = [
                SpecialTokens.BOS, SpecialTokens.EOS, SpecialTokens.UNK, SpecialTokens.PAD,
                SpecialTokens.ADDITIONAL, SpecialTokens.ROLE
            ]

        # Step 3: Mapping from Enum values to actual tokenizer attributes
        special_token_mapping = {
            SpecialTokens.BOS: self.model.tokenizer.bos_token_id,
            SpecialTokens.EOS: self.model.tokenizer.eos_token_id,
            SpecialTokens.UNK: self.model.tokenizer.unk_token_id,
            SpecialTokens.PAD: self.model.tokenizer.pad_token_id,
            SpecialTokens.ADDITIONAL: [self.model.tokenizer.convert_tokens_to_ids(token) 
                                    for token in self.model.tokenizer.additional_special_tokens],
            SpecialTokens.ROLE: [self.model.tokenizer.convert_tokens_to_ids(token) 
                                for token in ['user', 'model']] 
        }

        # Step 4: Collect the token IDs based on the selection
        selected_token_ids = []
        for token_enum in selected_special_tokens:
            token_value = special_token_mapping.get(token_enum)
            if token_value is not None:
                if isinstance(token_value, list):
                    selected_token_ids.extend(token_value)
                else:
                    selected_token_ids.append(token_value)

        # Step 5: Convert to tensor and return
        return torch.tensor(selected_token_ids, device=self.device)

    def get_special_tokens_mask(self, tokens, selected_special_tokens=None):
        """Return the special tokens tensor."""
        if selected_special_tokens is None:
            special_tokens = self.default_special_tokens_tensor
        else:
            special_tokens = self._get_special_tokens_tensor(selected_special_tokens)

        special_token_mask = torch.where(torch.isin(tokens, special_tokens), 1, 0)

        return special_token_mask

    @staticmethod
    def load_supported_dataset(dataset_name, split, local_dataset, base_folder_path):
        """Load a supported dataset from Hugging Face by name."""
        # if not isinstance(dataset_name, SupportedDatasets):
        #     raise ValueError(f"{dataset_name} is not a supported dataset. Choose from {list(SupportedDatasets)}")

        # handle CSV files
        if dataset_name.category == DatasetCategory.AGREEMENT_BE:
            if local_dataset:
                local_dataset_path = str(base_folder_path / dataset_name.value)
                return load_dataset('csv', data_files=local_dataset_path, split='train')
            else:
                raise ValueError(f"CSV datasets must be loaded locally. Set local_dataset=True.")

        if local_dataset:
            local_dataset_name = str(base_folder_path / dataset_name.value)
            return load_dataset('json', data_files=local_dataset_name, split='train')

        dataset_hf = load_dataset(dataset_name.value)  # Load dataset using HF datasets library
        if split not in dataset_hf:
            raise ValueError(f"Split '{split}' not found in the dataset. Available splits: {list(dataset_hf.keys())}")

        return dataset_hf[split]  # Return the selected split (e.g., 'train')
    
    def apply_chat_template_and_tokenize(self, prompt, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False):
        tokenizer = self.model.tokenizer
        special_token_mask = None

        # Apply chat template if required
        if apply_chat_template:
            conversation = [
                {"role": "user", "content": prompt}
            ]

            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                continue_final_message = False if prepend_generation_prefix else True,
                add_generation_prompt = prepend_generation_prefix
            )
        
        # Apply padding when manually tokenizing if needed
        if tokenize:
            # Check if the dataset category requires padding
            padding_strategy = 'max_length' if self.dataset_name.category.requires_padding else 'longest'
            max_length = self._max_prompt_length if self.dataset_name.category.requires_padding else None
            
            if apply_chat_template:
                # Tokenize using the tokenizer with appropriate padding strategy
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=padding_strategy,
                    truncation=True if max_length else False,
                    max_length=max_length,
                    return_special_tokens_mask=False
                )
                tokenized['input_ids'] = tokenized['input_ids'].to(self.device).squeeze(0)
                tokenized['special_tokens_mask'] = self.get_special_tokens_mask(tokenized['input_ids'])
            else:
                # Tokenize using the tokenizer with appropriate padding strategy, and add special tokens
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding=padding_strategy,
                    truncation=True if max_length else False,
                    max_length=max_length,
                    return_special_tokens_mask=True
                )
                
            prompt = tokenized["input_ids"].to(self.device).squeeze(0)
            special_token_mask = tokenized["special_tokens_mask"].squeeze(0).to(self.device)

        return prompt, special_token_mask

    def get_formatted_prompt(self, item, system_prompt, task_prompt, patched=False):
        if self.dataset_name.category == DatasetCategory.QA:
            choices = [
                f"{label}) {text}" 
                for label, text in zip(item['choices']['label'], item['choices']['text'])
            ]
            question_with_choices = f"{item['question']}\n" + "\n".join(choices)

            prompt = (
                f"{system_prompt} Now, here's the user's question:"
                f'\n"{question_with_choices}"'
                f'\n{task_prompt}'
            )
            return prompt
        elif self.dataset_name.category in [DatasetCategory.AGREEMENT, DatasetCategory.AGREEMENT_BE]:
            if not patched:
                return item['clean_prefix']
            else:
                return item['patch_prefix']
        elif self.dataset_name.category == DatasetCategory.TRUE_FALSE:
            question = f"'{item['statement']}' - Is this statement True or False?'"
            
            prompt = (
                f"{system_prompt} Now, here's the user's question:"
                f'\n"{question}"'
                f'\n{task_prompt}'
            )
            return prompt
        else:
            raise ValueError(f"Dataset category {self.dataset_name.category} not supported.")

    # For the get_clean_answer method:
    def get_clean_answer(self, item, prompt, tokenize=True):
        if self.dataset_name.category == DatasetCategory.QA:
            answer = item['answerKey']
        elif self.dataset_name.category == DatasetCategory.AGREEMENT:
            answer = item['clean_answer']
        elif self.dataset_name.category == DatasetCategory.AGREEMENT_BE:
            answer = item['be_conjugations_clean']
        elif self.dataset_name.category == DatasetCategory.TRUE_FALSE:
            answer = str(item['label'])
        else:
            raise ValueError(f"Dataset category {self.dataset_name.category} not supported.")

        try:
            # Find answer pos as the first token before padding in the prompt
            answer_pos = find_first_index(prompt, self.model.tokenizer.pad_token_id) - 1
        except ValueError: # If this doesn't work, either the prompt is not tokenizer or it's too long
            # In which case it's enough to provide the last token position
            answer_pos = prompt.shape[0] - 1

        if tokenize:
            answer = self.model.to_single_token(answer)

        return answer, answer_pos

    # For the get_corrupted_answer method:
    def get_corrupted_answer(self, item, prompt, tokenize=True):
        if self.dataset_name.category == DatasetCategory.QA:
            correct_answer = item['answerKey']
            answer = [option for option in ['A', 'B', 'C', 'D', 'E'] if option != correct_answer]
        elif self.dataset_name.category == DatasetCategory.AGREEMENT:
            answer = item['patch_answer']
        elif self.dataset_name.category == DatasetCategory.AGREEMENT_BE:
            answer = item['be_conjugations_patch']
        elif self.dataset_name.category == DatasetCategory.TRUE_FALSE:
            answer = str(not item['label'])
        else:
            raise ValueError(f"Dataset category {self.dataset_name.category} not supported.")

        try:
            # Find answer pos as the first token before padding in the prompt
            answer_pos = find_first_index(prompt, self.model.tokenizer.pad_token_id) - 1
        except ValueError: # If this doesn't work, either the prompt is not tokenizer or it's too long
            # In which case it's enough to provide the last token position
            answer_pos = prompt.shape[0] - 1

        if tokenize:
            if isinstance(answer, list):
                answer = [self.model.to_single_token(option) for option in answer]
            else:
                answer = self.model.to_single_token(answer)

        return answer, answer_pos

    def get_masks(self, special_token_mask):
        # Find the first non-special token index
        non_special_indices = torch.nonzero(special_token_mask == 0, as_tuple=False)
        if len(non_special_indices) > 0:
            first_non_special_token_idx = non_special_indices[0].item()
        else:
            # If no non-special tokens found, use the first token
            first_non_special_token_idx = 0
        
        control_sequence_length = first_non_special_token_idx + 1

        # Create attention mask (1 for tokens to attend to, 0 for padding tokens)
        attention_mask = torch.ones_like(special_token_mask)
        
        # Try to find padding tokens from the end
        reversed_special_mask = torch.flip(special_token_mask, [0])
        reversed_indices = torch.nonzero(reversed_special_mask == 0, as_tuple=False)
        
        if len(reversed_indices) > 0:
            # Found non-special tokens from the end
            first_padding_pos_reversed = reversed_indices[0].item() - 1
            if first_padding_pos_reversed >= 0:  # Ensure it's a valid position
                first_padding_pos = len(special_token_mask) - 1 - first_padding_pos_reversed
                attention_mask[first_padding_pos:] = 0
        
        return control_sequence_length, attention_mask
            
    def get_clean_sample(self, item, tokenize, apply_chat_template, prepend_generation_prefix=False):
        """Process each example from the dataset with padding when tokenizing."""

        prompt = self.get_formatted_prompt(item, system_prompt=self.clean_system_prompt, task_prompt=self.task_prompt, patched=False)

        prompt, special_token_mask = self.apply_chat_template_and_tokenize(prompt, tokenize=tokenize, 
                                                                           apply_chat_template=apply_chat_template, 
                                                                           prepend_generation_prefix=prepend_generation_prefix)

        # Construct the answer key
        clean_answer, clean_answer_pos = self.get_clean_answer(item, prompt, tokenize=tokenize)

        # Prepare the result dictionary
        result_dict = {
            "prompt": prompt,
            "answer": clean_answer,
            "answer_pos": clean_answer_pos
        }

        # Include special_token_mask if tokenization was applied
        if tokenize:
            result_dict["special_token_mask"] = special_token_mask

            control_sequence_length, attention_mask = self.get_masks(special_token_mask)
            result_dict["control_sequence_length"] = control_sequence_length
            result_dict["attention_mask"] = attention_mask

        return result_dict

    def get_corrupted_sample(self, item, tokenize, apply_chat_template, prepend_generation_prefix=False):
        """Process each example from the dataset with padding when tokenizing."""

        prompt = self.get_formatted_prompt(item, system_prompt=self.corrupted_system_prompt, task_prompt=self.task_prompt, patched=True)

        prompt, special_token_mask = self.apply_chat_template_and_tokenize(prompt, tokenize=tokenize, 
                                                                           apply_chat_template=apply_chat_template, 
                                                                           prepend_generation_prefix=prepend_generation_prefix)
        
        # Construct the answer key
        corrupted_answer, corrupted_answer_pos = self.get_corrupted_answer(item, prompt, tokenize=tokenize)

        # Prepare the result dictionary
        result_dict = {
            "prompt": prompt,
            "answer": corrupted_answer,
            "answer_pos": corrupted_answer_pos
        }

        # Include special_token_mask if tokenization was applied
        if tokenize:
            result_dict["special_token_mask"] = special_token_mask

            control_sequence_length, attention_mask = self.get_masks(special_token_mask)
            result_dict["control_sequence_length"] = control_sequence_length
            result_dict["attention_mask"] = attention_mask

        return result_dict

    def get_single_dataset(self, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False, pt=True):
        """
        Processes datasets of format {"prompt": "...", "true_answer": "...", "false_answer": "..."}.
        This method is designed for datasets that do not have a clean/corrupted split.

        Args:
            tokenize (bool): Whether to tokenize the prompts.
            apply_chat_template (bool): Whether to apply chat template.
            prepend_generation_prefix (bool): Whether to prepend generation prefix.
            pt (bool): Whether to return PyTorch tensors (True) or a HuggingFace dataset (False).
        """
        if self.dataset_name.category.value != DatasetCategory.PLAIN_PROMPT.value:
            raise ValueError(f"This method is only for PLAIN_PROMPT category datasets, but got {self.dataset_name.category}")

        if self.dataset_name.category.requires_padding:
            print('Figuring out optimal padding length...')
            self.filter_and_set_max_length(apply_chat_template=apply_chat_template,
                                           prepend_generation_prefix=prepend_generation_prefix)
        else:
            print(f'Dataset {self.dataset_name} has constant length prompts. Skipping padding calculation.')

        samples = []
        for item in tqdm(self.dataset):
            prompt_text = item['prompt']
            prompt, special_token_mask = self.apply_chat_template_and_tokenize(
                prompt_text,
                tokenize=tokenize,
                apply_chat_template=apply_chat_template,
                prepend_generation_prefix=prepend_generation_prefix
            )

            true_answer_text = item['true_answer']
            false_answer_text = item['false_answer']

            try:
                answer_pos = find_first_index(prompt, self.model.tokenizer.pad_token_id) - 1
            except ValueError:
                answer_pos = prompt.shape[0] - 1

            true_answer = true_answer_text
            false_answer = false_answer_text
            if tokenize:
                true_answer = self.model.to_single_token(true_answer_text)
                false_answer = self.model.to_single_token(false_answer_text)

            sample = {
                "prompt": prompt,
                "true_answer": true_answer,
                "false_answer": false_answer,
                "answer_pos": answer_pos,
            }

            if tokenize:
                sample["special_token_mask"] = special_token_mask
                control_sequence_length, attention_mask = self.get_masks(special_token_mask)
                sample["control_sequence_length"] = control_sequence_length
                sample["attention_mask"] = attention_mask

            samples.append(sample)

        hf_dict = {
            "prompt": [entry["prompt"] for entry in samples],
            "true_answer": [entry["true_answer"] for entry in samples],
            "false_answer": [entry["false_answer"] for entry in samples],
            "answer_pos": [entry["answer_pos"] for entry in samples]
        }

        if tokenize:
            hf_dict["special_token_mask"] = [entry.get("special_token_mask", torch.zeros_like(entry["prompt"], device=self.device)) for entry in samples]
            hf_dict["control_sequence_length"] = [entry.get("control_sequence_length", 0) for entry in samples]
            hf_dict["attention_mask"] = [entry.get("attention_mask", torch.ones_like(entry["prompt"], device=self.device)) for entry in samples]

        if pt:
            hf_dict['prompt'] = torch.stack(hf_dict['prompt'])
            hf_dict['true_answer'] = torch.tensor(hf_dict['true_answer'], device=self.device)
            hf_dict['false_answer'] = torch.tensor(hf_dict['false_answer'], device=self.device)
            hf_dict['answer_pos'] = torch.tensor(hf_dict['answer_pos'], device=self.device)

            if tokenize:
                hf_dict['special_token_mask'] = torch.stack(hf_dict['special_token_mask'])
                hf_dict['control_sequence_length'] = torch.tensor(hf_dict['control_sequence_length'], device=self.device)
                hf_dict['attention_mask'] = torch.stack(hf_dict['attention_mask'])
            
            return hf_dict
        else:
            return Dataset.from_dict(hf_dict)

    def save_processed_subset(
        self, 
        eval_indices: list, 
        batch_size: int, 
        clean_dataset: dict, 
        patched_dataset: dict, 
        filename_suffix: str
    ):
        """
        Creates and saves a new dataset containing only the specific clean or
        patched runs that match the provided evaluation indices.

        This version operates on the final tokenized datasets and correctly
        assigns the "true" and "false" answers for each run. The saved file
        contains detokenized text, ready for the get_single_dataset method.

        Args:
            eval_indices (list): A list of flat indices from the 2*N evaluation tensor.
            batch_size (int): The batch size used during the original evaluation.
            clean_dataset (dict): The final, tokenized clean dataset dictionary.
            patched_dataset (dict): The final, tokenized patched dataset dictionary.
            filename_suffix (str): A descriptive suffix for the output filename.
        """
        print(f"Processing {len(eval_indices)} selected evaluation runs to create new dataset...")

        # Step 1: Build the definitive map from evaluation index to source index and run type.
        # This correctly handles the final, incomplete batch.
        source_dataset_size = len(clean_dataset['prompt'])
        n_full_batches = source_dataset_size // batch_size
        last_batch_size = source_dataset_size % batch_size
        total_batches = n_full_batches + (1 if last_batch_size > 0 else 0)

        eval_to_source_map = {}
        flat_cursor, source_cursor = 0, 0
        
        for i in range(total_batches):
            is_last_batch = (i == total_batches - 1) and (last_batch_size > 0)
            current_batch_size = last_batch_size if is_last_batch else batch_size

            for j in range(current_batch_size):
                eval_to_source_map[flat_cursor + j] = (source_cursor + j, False) # False for clean run
            flat_cursor += current_batch_size
            
            for j in range(current_batch_size):
                eval_to_source_map[flat_cursor + j] = (source_cursor + j, True)  # True for patched run
            flat_cursor += current_batch_size
            
            source_cursor += current_batch_size
            
        # Step 2: Process the selected indices and create the new samples.
        processed_samples = []
        eval_indices_set = set(eval_indices)

        for flat_index in tqdm(eval_indices, desc="Processing samples"):
            if flat_index not in eval_indices_set:
                continue
                
            source_index, is_patched_run = eval_to_source_map[flat_index]

            # --- Get the prompt tensor from the correct source ---
            prompt_tensor = patched_dataset['prompt'][source_index] if is_patched_run else clean_dataset['prompt'][source_index]
            
            # --- Get the answer tokens from both sources ---
            clean_answer_token = clean_dataset['answer'][source_index]
            patched_answer_token = patched_dataset['answer'][source_index]
            
            # --- Detokenize everything to text for saving ---
            # Filter out padding tokens before converting to string
            pad_token_id = self.model.tokenizer.pad_token_id
            bos_token_id = self.model.tokenizer.bos_token_id
            special_tokens = torch.tensor([pad_token_id, bos_token_id], device=self.device)

            prompt_text = self.model.to_string(prompt_tensor[~torch.isin(prompt_tensor, special_tokens)])
            
            # Detokenize single answer tokens
            clean_answer_text = self.model.to_string(clean_answer_token)
            patched_answer_text = self.model.to_string(patched_answer_token)
            
            # --- Correctly assign "true" and "false" answers based on the run type ---
            if is_patched_run:
                true_answer = patched_answer_text
                false_answer = clean_answer_text
            else: # is_clean_run
                true_answer = clean_answer_text
                false_answer = patched_answer_text
                
            # Assemble the new sample in the target format
            processed_samples.append({
                "prompt": prompt_text,
                "true_answer": true_answer,
                "false_answer": false_answer
            })
            
        # Step 3: Create and save the new Hugging Face Dataset
        if not processed_samples:
            print("Warning: No samples were processed. No file will be saved.")
            return
            
        new_dataset = Dataset.from_list(processed_samples)

        file_name_prefix = self.dataset_name.value.replace('/', '_')
        file_name = f"{file_name_prefix}{filename_suffix}.json"
        
        output_path = Path(self.base_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / file_name

        print(f"Saving new processed dataset with {len(new_dataset)} samples to: {full_path}")
        new_dataset.to_json(full_path)
        print("Save complete.")

    def get_clean_corrupted_datasets(self, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False, 
                                     pt=True, filter_long_sequences=True, use_most_common_length=False, save_filtered_dataset=False):
        """
        Refactored method to return PyTorch tensors if the 'pt' parameter is set to True (default).
        For categories that require padding (QA, TRUE_FALSE), computes optimal padding.
        For categories with constant lengths (AGREEMENT, AGREEMENT_BE), no padding is applied.
        Always returns a consistent set of fields regardless of dataset category.

        Args:
            tokenize (bool): Whether to tokenize the prompts
            apply_chat_template (bool): Whether to apply chat template
            prepend_generation_prefix (bool): Whether to prepend generation prefix
            pt (bool): Whether to return PyTorch tensors (True) or HuggingFace datasets (False)
            filter_long_sequences (bool): Whether to filter out sequences longer than 99th percentile
            use_most_common_length (bool): Whether to filter to only the most common prompt length
            save_filtered_dataset (bool): Whether to save the filtered dataset to a new JSON file
        """
        if self.dataset_name.category.requires_padding:
            print('Figuring out optimal padding length...')
            self.filter_and_set_max_length(apply_chat_template=apply_chat_template, 
                                         prepend_generation_prefix=prepend_generation_prefix,
                                         filter_long_sequences=filter_long_sequences,
                                         use_most_common_length=use_most_common_length)
            
            if use_most_common_length and save_filtered_dataset:
                file_name_prefix = self.dataset_name.value.replace('/', '_')
                file_name = f"{file_name_prefix}_constant_length.json"
                if not os.path.exists(self.base_folder_path):
                    os.makedirs(self.base_folder_path)
                full_path = os.path.join(self.base_folder_path, file_name)
                self.dataset.to_json(full_path)
                print(f"Saved dataset with constant length prompts to {full_path}")
        else:
            print(f'Dataset {self.dataset_name} has constant length prompts. Skipping padding calculation.')

        clean_samples = []
        corrupted_samples = []

        for item in tqdm(self.dataset):
            # Process the example
            clean_sample = self.get_clean_sample(item, tokenize=tokenize, prepend_generation_prefix=prepend_generation_prefix,
                                                apply_chat_template=apply_chat_template)
            corrupted_sample = self.get_corrupted_sample(item, tokenize=tokenize, prepend_generation_prefix=prepend_generation_prefix,
                                                        apply_chat_template=apply_chat_template)

            clean_samples.append(clean_sample)
            corrupted_samples.append(corrupted_sample)

        # Convert list of dictionaries into dictionaries suitable for Hugging Face Datasets or PyTorch tensors
        clean_hf_dict = {
            "prompt": [entry["prompt"] for entry in clean_samples],
            "answer": [entry["answer"] for entry in clean_samples],
            "answer_pos": [entry["answer_pos"] for entry in clean_samples]
        }
        corrupted_hf_dict = {
            "prompt": [entry["prompt"] for entry in corrupted_samples],
            "answer": [entry["answer"] for entry in corrupted_samples],
            "answer_pos": [entry["answer_pos"] for entry in corrupted_samples]
        }

        # Add special_token_mask if tokenization was applied
        if tokenize:
            clean_hf_dict["special_token_mask"] = [entry.get("special_token_mask", torch.zeros_like(entry["prompt"], device=self.device)) for entry in clean_samples]
            corrupted_hf_dict["special_token_mask"] = [entry.get("special_token_mask", torch.zeros_like(entry["prompt"], device=self.device)) for entry in corrupted_samples]

            # Add control_sequence_length with default of 0 if not present
            clean_hf_dict["control_sequence_length"] = [entry.get("control_sequence_length", 0) for entry in clean_samples]
            corrupted_hf_dict["control_sequence_length"] = [entry.get("control_sequence_length", 0) for entry in corrupted_samples]
            
            # Add attention_mask with default of all 1s if not present
            clean_hf_dict["attention_mask"] = [entry.get("attention_mask", torch.ones_like(entry["prompt"], device=self.device)) for entry in clean_samples]
            corrupted_hf_dict["attention_mask"] = [entry.get("attention_mask", torch.ones_like(entry["prompt"], device=self.device)) for entry in corrupted_samples]

        if pt:
            for dataset_dict in [clean_hf_dict, corrupted_hf_dict]:
                dataset_dict['prompt'] = torch.stack(dataset_dict['prompt'])
                dataset_dict['answer'] = torch.tensor(dataset_dict['answer'], device=self.device)
                dataset_dict['answer_pos'] = torch.tensor(dataset_dict['answer_pos'], device=self.device)

                if tokenize:
                    dataset_dict['special_token_mask'] = torch.stack(dataset_dict['special_token_mask'])
                    dataset_dict['control_sequence_length'] = torch.tensor(dataset_dict['control_sequence_length'], device=self.device)
                    dataset_dict['attention_mask'] = torch.stack(dataset_dict['attention_mask'])

            return clean_hf_dict, corrupted_hf_dict
        else:
            # If not using PyTorch tensors, return as Hugging Face Datasets
            clean_hf_dataset = Dataset.from_dict(clean_hf_dict)
            corrupted_hf_dataset = Dataset.from_dict(corrupted_hf_dict)

            return clean_hf_dataset, corrupted_hf_dataset