from enum import Enum

# Enum listing special Gemma-2 tokens
class SpecialTokens(Enum):
    BOS = 'bos_token'
    EOS = 'eos_token'
    UNK = 'unk_token'
    PAD = 'pad_token'   
    ADDITIONAL = 'additional_special_tokens' # start_of_turn, end_of_turn
    ROLE = 'role_tokens' # user, model

# Enum to define dataset categories
class DatasetCategory(Enum):
    QA = "qa"
    AGREEMENT = "agreement"
    TRUE_FALSE = "true_false"

# Refactored enum listing supported datasets with categories
class SupportedDatasets(Enum):
    COMMONSENSE_QA = ("tau/commonsense_qa", DatasetCategory.QA)
    COMMONSENSE_QA_FILTERED = ("drsis/deception-commonsense_qa_wo_chat", DatasetCategory.QA)
    VERB_AGREEMENT = ('rc_train_filtered.json', DatasetCategory.AGREEMENT)
    VERB_AGREEMENT_TEST = ('rc_test.json', DatasetCategory.AGREEMENT)
    CITIES = ('cities_true_false.json', DatasetCategory.TRUE_FALSE)
    COMPANIES = ('companies_true_false.json', DatasetCategory.TRUE_FALSE)
    FACTS = ('facts_true_false.json', DatasetCategory.TRUE_FALSE)
    
    def __init__(self, path, category):
        self.path = path
        self.category = category
    
    @property
    def value(self):
        # For backward compatibility with existing code that expects value to be the path
        return self.path