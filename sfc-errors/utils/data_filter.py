# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
project_dir = Path(__file__).parent.parent.parent

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="cuda",
)


#%%
def process_data_in_batches(rc_data: pd.DataFrame, model, tokenizer, batch_size: int = 128) -> pd.DataFrame:
    """
    Process data in batches and add prediction metrics to the dataframe.
    
    Args:
        rc_data: DataFrame containing the data to process
        model: The language model to use for predictions
        tokenizer: The tokenizer for the model
        batch_size: Size of batches to process
        
    Returns:
        DataFrame with added prediction metrics
    """
    result_df = rc_data.copy()
    
    # Add columns to store results
    result_columns = [
        'clean_answer_logits', 'clean_answer_probs', 'clean_answer_rank',
        'clean_be_conjugations_rank', 'clean_be_conjugations_logits_diff',
        'clean_be_conjugations_probs', 'clean_be_conjugations_probs_diff',
        'patch_answer_logits', 'patch_answer_probs', 'patch_answer_rank',
        'patch_be_conjugations_rank', 'patch_be_conjugations_logits_diff',
        'patch_be_conjugations_probs', 'patch_be_conjugations_probs_diff'
    ]
    
    for col in result_columns:
        result_df[col] = None
    
    num_samples = len(rc_data)
    
    for start_idx in tqdm(range(0, num_samples, batch_size)):
        if (start_idx / batch_size) % 5 == 0:
            t.cuda.empty_cache()
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch = slice(start_idx, end_idx)
        
        # Tokenize inputs
        clean_toks = tokenizer(rc_data["clean_prefix"].iloc[current_batch].to_list(), 
                              return_tensors="pt", padding=True).to("cuda")
        clean_answers = tokenizer(rc_data["clean_answer"].iloc[current_batch].to_list(), 
                                 return_tensors="np", add_special_tokens=False)['input_ids'].flatten().tolist()
        patch_toks = tokenizer(rc_data["patch_prefix"].iloc[current_batch].to_list(), 
                              return_tensors="pt", padding=True).to("cuda")
        patch_answers = tokenizer(rc_data["patch_answer"].iloc[current_batch].to_list(), 
                                 return_tensors="np", add_special_tokens=False)['input_ids'].flatten().tolist()
        be_conjugations_clean_toks = rc_data['be_conjugations_clean_tok'].iloc[current_batch].to_list()
        be_conjugations_patch_toks = rc_data['be_conjugations_patch_tok'].iloc[current_batch].to_list()
        
        # Get model predictions
        with t.no_grad():
            clean_out = model.forward(**clean_toks)
            patch_out = model.forward(**patch_toks)
        
        # Process clean outputs
        clean_logits = clean_out.logits[:,-1,:]
        clean_probs = t.nn.functional.softmax(clean_logits, dim=-1)
        clean_logits_rank = t.argsort(clean_logits, dim=-1, descending=True)
        
        patch_logits = patch_out.logits[:,-1,:]
        patch_probs = t.nn.functional.softmax(patch_logits, dim=-1)
        patch_logits_rank = t.argsort(patch_logits, dim=-1, descending=True)
        
        # We need to update each row individually
        for i, idx in enumerate(range(start_idx, end_idx)):
            # Clean metrics
            result_df.at[idx, 'clean_answer_logits'] = clean_logits[i, clean_answers[i]].item()
            result_df.at[idx, 'clean_answer_probs'] = clean_probs[i, clean_answers[i]].item()
            result_df.at[idx, 'clean_answer_rank'] = t.where(clean_logits_rank[i] == clean_answers[i])[0].item()
            result_df.at[idx, 'clean_be_conjugations_rank'] = t.where(clean_logits_rank[i] == be_conjugations_clean_toks[i])[0].item()
            result_df.at[idx, 'clean_be_conjugations_logits_diff'] = (
                clean_logits[i, be_conjugations_clean_toks[i]] - 
                clean_logits[i, be_conjugations_patch_toks[i]]).item()
            result_df.at[idx, 'clean_be_conjugations_probs'] = clean_probs[i, be_conjugations_clean_toks[i]].item()
            result_df.at[idx, 'clean_be_conjugations_probs_diff'] = (
                clean_probs[i, be_conjugations_clean_toks[i]] - 
                clean_probs[i, be_conjugations_patch_toks[i]]).item()
            
            # Patch metrics
            result_df.at[idx, 'patch_answer_logits'] = patch_logits[i, patch_answers[i]].item()
            result_df.at[idx, 'patch_answer_probs'] = patch_probs[i, patch_answers[i]].item()
            result_df.at[idx, 'patch_answer_rank'] = t.where(patch_logits_rank[i] == patch_answers[i])[0].item()
            result_df.at[idx, 'patch_be_conjugations_rank'] = t.where(patch_logits_rank[i] == be_conjugations_patch_toks[i])[0].item()
            result_df.at[idx, 'patch_be_conjugations_logits_diff'] = (
                patch_logits[i, be_conjugations_patch_toks[i]] - 
                patch_logits[i, be_conjugations_clean_toks[i]]).item()
            result_df.at[idx, 'patch_be_conjugations_probs'] = patch_probs[i, be_conjugations_patch_toks[i]].item()
            result_df.at[idx, 'patch_be_conjugations_probs_diff'] = (
                patch_probs[i, be_conjugations_patch_toks[i]] - 
                patch_probs[i, be_conjugations_clean_toks[i]]).item()
    
    return result_df
# %%


if __name__ == "__main__":
    rc_test = pd.read_json(project_dir / "data" / "rc_test.json", lines=True)
    rc_train = pd.read_json(project_dir / "data" / "rc_train_filtered.json", lines=True)


    rc_test['be_conjugations_clean'] = np.where(rc_test['case'].str[:4] == 'plur', ' are', ' is')
    rc_test['be_conjugations_clean_tok'] = tokenizer(rc_test['be_conjugations_clean'].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()
    rc_test['be_conjugations_patch'] = np.where(rc_test['case'].str[:4] == 'plur', ' is', ' are')
    rc_test['be_conjugations_patch_tok'] = tokenizer(rc_test['be_conjugations_patch'].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()

    rc_train['be_conjugations_clean'] = np.where(rc_train['case'].str[:4] == 'plur', ' are', ' is')
    rc_train['be_conjugations_clean_tok'] = tokenizer(rc_train['be_conjugations_clean'].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()
    rc_train['be_conjugations_patch'] = np.where(rc_train['case'].str[:4] == 'plur', ' is', ' are')
    rc_train['be_conjugations_patch_tok'] = tokenizer(rc_train['be_conjugations_patch'].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()

    processed_rc_test = process_data_in_batches(rc_test, model, tokenizer)
    processed_rc_train = process_data_in_batches(rc_train, model, tokenizer)

    processed_rc_test.to_csv(project_dir / "data" / "rc_test_processed.csv", index=False)
    processed_rc_train.to_csv(project_dir / "data" / "rc_train_processed.csv", index=False)





"""
Old code

clean_toks = tokenizer(rc_data["clean_prefix"].iloc[0:10].to_list(), return_tensors="pt", padding = True).to("cuda")
clean_answers = tokenizer(rc_data["clean_answer"].iloc[0:10].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()
patch_toks = tokenizer(rc_data["patch_prefix"].iloc[0:10].to_list(), return_tensors="pt", padding = True).to("cuda")
patch_answers = tokenizer(rc_data["patch_answer"].iloc[0:10].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()
be_conjugations_clean_toks = tokenizer(rc_data['be_conjugations_clean'].iloc[0:10].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()
be_conjugations_patch_toks = tokenizer(rc_data['be_conjugations_patch'].iloc[0:10].to_list(), return_tensors = "np", add_special_tokens=False)['input_ids'].flatten().tolist()



with t.no_grad():
    clean_out = model.forward(**clean_toks)
    patch_out = model.forward(**patch_toks)

clean_logits = clean_out.logits[:,-1,:]
clean_probs = t.nn.functional.softmax(clean_logits, dim=-1)
clean_logits_rank = t.argsort(clean_logits, dim=-1, descending=True)
clean_answer_logits = clean_logits[t.arange(clean_logits.size(0)), clean_answers]
clean_answer_probs = clean_probs[t.arange(clean_probs.size(0)), clean_answers]
clean_answer_rank = clean_logits_rank[t.arange(clean_logits_rank.size(0)), clean_answers]
clean_be_conjugations_rank = clean_logits_rank[t.arange(clean_logits_rank.size(0)), be_conjugations_clean_toks]
clean_be_conjugations_logits_diff = clean_logits[t.arange(clean_logits.size(0)), be_conjugations_clean_toks] - clean_logits[t.arange(clean_logits.size(0)), be_conjugations_patch_toks]
clean_be_conjugations_probs = clean_probs[t.arange(clean_probs.size(0)), be_conjugations_clean_toks]
clean_be_conjugations_probs_diff = clean_be_conjugations_probs - clean_probs[t.arange(clean_probs.size(0)), be_conjugations_patch_toks]

patch_logits = patch_out.logits[:,-1,:]
patch_probs = t.nn.functional.softmax(patch_logits, dim=-1)
patch_logits_rank = t.argsort(patch_logits, dim=-1, descending=True)
patch_answer_logits = patch_logits[t.arange(patch_logits.size(0)), patch_answers]
patch_answer_probs = patch_probs[t.arange(patch_probs.size(0)), patch_answers]
patch_answer_rank = patch_logits_rank[t.arange(patch_logits_rank.size(0)), patch_answers]
patch_be_conjugations_rank = patch_logits_rank[t.arange(patch_logits_rank.size(0)), be_conjugations_patch_toks]
patch_be_conjugations_logits_diff = patch_logits[t.arange(patch_logits.size(0)), be_conjugations_patch_toks] - patch_logits[t.arange(patch_logits.size(0)), be_conjugations_clean_toks]
patch_be_conjugations_probs = patch_probs[t.arange(patch_probs.size(0)), be_conjugations_patch_toks]
patch_be_conjugations_probs_diff = patch_be_conjugations_probs - patch_probs[t.arange(patch_probs.size(0)), be_conjugations_clean_toks]
"""