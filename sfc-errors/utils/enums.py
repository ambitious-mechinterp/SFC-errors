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
    QA = ("qa", True)
    AGREEMENT = ("agreement", False)
    AGREEMENT_BE = ("agreement_be", True)
    TRUE_FALSE = ("true_false", True)
    PLAIN_PROMPT = ("plain_prompt", False)
    
    def __init__(self, value, requires_padding):
        # We use the first element as the value
        self._value_ = value
        # Store requires_padding as a separate attribute
        self._requires_padding = requires_padding
    
    @property
    def requires_padding(self):
        return self._requires_padding

# Refactored enum listing supported datasets with categories
class SupportedDatasets(Enum):
    COMMONSENSE_QA = ("tau/commonsense_qa", DatasetCategory.QA)
    COMMONSENSE_QA_FILTERED = ("drsis/deception-commonsense_qa_wo_chat", DatasetCategory.QA)
    VERB_AGREEMENT = ('rc_train_constant_length.json', DatasetCategory.AGREEMENT)

    VERB_AGREEMENT_TEST = ('rc_test_constant_length.json', DatasetCategory.AGREEMENT)
    VERB_AGREEMENT_TEST_MM_FILTERED = ('rc_test_mm_gt_4.json', DatasetCategory.AGREEMENT)

    VERB_AGREEMENT_BE = ('rc_train_processed.csv', DatasetCategory.AGREEMENT_BE)
    VERB_AGREEMENT_TEST_BE = ('rc_test_processed.csv', DatasetCategory.AGREEMENT_BE)
    CITIES = ('cities_true_false.json', DatasetCategory.TRUE_FALSE)
    COMPANIES = ('companies_true_false.json', DatasetCategory.TRUE_FALSE)
    FACTS = ('facts_true_false.json', DatasetCategory.TRUE_FALSE)
    VERB_AGREEMENT_TEST_CONFIDENT_MODEL = ('rc_test_confident_model.json', DatasetCategory.PLAIN_PROMPT)
    VERB_AGREEMENT_TEST_CONFIDENT_MODEL_SALIENT_CIRCUIT = \
        ('rc_test_confident_model_salient_circuit.json', DatasetCategory.PLAIN_PROMPT)
    
    def __init__(self, path, category):
        # We use the path as the value
        self._value_ = path
        # Store category as a separate attribute
        self._category = category
    
    @property
    def path(self):
        return self._value_
    
    @property
    def category(self):
        return self._category