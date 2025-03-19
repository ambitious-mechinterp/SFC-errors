from datasets import load_dataset, Dataset
from utils.enums import SpecialTokens, SupportedDatasets, DatasetCategory
import torch
import random
from tqdm import tqdm

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

    # Old & awful version of setting max prompt length
    # def _get_max_prompt_length(self):
    #     if self.dataset_name in [SupportedDatasets.COMMONSENSE_QA, SupportedDatasets.COMMONSENSE_QA_FILTERED]:
    #         return 180
    #     elif self.dataset_name == SupportedDatasets.VERB_AGREEMENT:
    #         return 30
    #     elif self.dataset_name in [SupportedDatasets.CITIES, SupportedDatasets.COMPANIES, SupportedDatasets.FACTS]:
    #         return 180

    def filter_and_set_max_length(self, apply_chat_template=True, prepend_generation_prefix=False):
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

        if local_dataset:
            local_dataset_name = str(base_folder_path / dataset_name.value)
            return load_dataset('json', data_files=local_dataset_name, split='train')

        dataset_hf = load_dataset(dataset_name.value)  # Load dataset using HF datasets library
        if split not in dataset_hf:
            raise ValueError(f"Split '{split}' not found in the dataset. Available splits: {list(dataset_hf.keys())}")

        return dataset_hf[split]  # Return the selected split (e.g., 'train')
    
    def apply_chat_template_and_tokenize(self, prompt, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False):
        tokenizer = self.model.tokenizer
        max_length = self._max_prompt_length
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
        
        # Apply padding when manually tokenizing
        if tokenize:
            if apply_chat_template:
                # Tokenize using the tokenizer with padding and truncation
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding='max_length',        # Pad to max_length
                    truncation=True,             # Truncate to max_length
                    max_length=max_length,
                    return_special_tokens_mask=False
                )
                tokenized['input_ids'] = tokenized['input_ids'].to(self.device).squeeze(0)
                tokenized['special_tokens_mask'] = self.get_special_tokens_mask(tokenized['input_ids'])
            else:
                # Tokenize using the tokenizer with padding and truncation, and add special tokens
                tokenized = tokenizer(
                    prompt, 
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding='max_length',        # Pad to max_length
                    truncation=True,             # Truncate to max_length
                    max_length=max_length,
                    return_special_tokens_mask=True
                )
                
            prompt = tokenized["input_ids"].to(self.device).squeeze(0)  # Padded input IDs
            special_token_mask = tokenized["special_tokens_mask"].squeeze(0).to(self.device)  # Mask for special tokens

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
        elif self.dataset_name.category == DatasetCategory.AGREEMENT:
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
        first_non_special_token_idx = torch.nonzero(special_token_mask == 0, as_tuple=False)[0].item()
        control_sequence_length = first_non_special_token_idx + 1

        attention_mask = torch.ones_like(special_token_mask)
        reversed_special_mask = torch.flip(special_token_mask, [0])

        first_padding_pos_reversed = torch.nonzero(reversed_special_mask == 0, as_tuple=False)[0].item() - 1
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

    def get_clean_corrupted_datasets(self, tokenize=True, apply_chat_template=True, prepend_generation_prefix=False, pt=True):
        """
        Refactored method to return PyTorch tensors if the 'pt' parameter is set to True (default).
        """
        print('Figuring out optimal padding length...')
        self.filter_and_set_max_length(apply_chat_template=apply_chat_template, prepend_generation_prefix=prepend_generation_prefix)

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

        if tokenize:
            clean_hf_dict["special_token_mask"] = [entry["special_token_mask"] for entry in clean_samples]
            corrupted_hf_dict["special_token_mask"] = [entry["special_token_mask"] for entry in corrupted_samples]

            clean_hf_dict["control_sequence_length"] = [entry["control_sequence_length"] for entry in clean_samples]
            corrupted_hf_dict["control_sequence_length"] = [entry["control_sequence_length"] for entry in corrupted_samples]

            clean_hf_dict["attention_mask"] = [entry["attention_mask"] for entry in clean_samples]
            corrupted_hf_dict["attention_mask"] = [entry["attention_mask"] for entry in corrupted_samples]

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