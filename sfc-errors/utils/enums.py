from enum import Enum

# Enum listing special Gemma-2 tokens
class SpecialTokens(Enum):
    BOS = 'bos_token'
    EOS = 'eos_token'
    UNK = 'unk_token'
    PAD = 'pad_token'   
    ADDITIONAL = 'additional_special_tokens' # start_of_turn, end_of_turn
    ROLE = 'role_tokens' # user, model

# Enum listing supported datasets
class SupportedDatasets(Enum):
    COMMONSENSE_QA = "tau/commonsense_qa"
    COMMONSENSE_QA_FILTERED = "drsis/deception-commonsense_qa_wo_chat"
    VERB_AGREEMENT = 'rc_train_filtered.json'
    CITIES = 'cities_true_false.json'
    COMPANIES = 'companies_true_false.json'
    FACTS = 'facts_true_false.json'