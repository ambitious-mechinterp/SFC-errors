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
    """
    Custom enum to define dataset categories and whether they require padding.
    A dataset category is a label for a group of datasets that share similar structure.
    For example, all verb-agreement datasets from the SFC paper can be grouped under the "agreement" category,
    which will hint our SFCDatasetLoader class to process them in a same way.

    Attributes:
        value (str): The string value representing the dataset category.
        requires_padding (bool): Indicates if datasets in this category require padding during tokenization
        (should be true if datasample are of different length, false otherwise).
    """
    QA = ("qa", True) # CommonsenseQA-like datasets
    AGREEMENT = ("agreement", False) # Verb-agreement datasets from SFC paper
    AGREEMENT_BE = ("agreement_be", True) # Our custom verb-agreement datasets with be-conjugations
    TRUE_FALSE = ("true_false", True) # True/False datasets
    PLAIN_PROMPT = ("plain_prompt", False) # Datasets that have a plain <prompt, true_answer, false_answer> structure
    
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
    """
    Enum listing supported datasets along with their categories.
    Attributes:
        path (str): The path to the dataset. Can be a name of the local file stored in the `data/` folder or a
                    path to a dataset on the HuggingFace hub.
        category (DatasetCategory): The category of the dataset, as defined above.
    """
    # Below starts the section with SFC datasets stored locally in the `data/` folder.
    # First, there are original versions that we've just filtered to be constant in length.

    VERB_AGREEMENT = ('rc_train_constant_length.json', DatasetCategory.AGREEMENT) 
    VERB_AGREEMENT_TEST = ('rc_test_constant_length.json', DatasetCategory.AGREEMENT)

    # Next there are our custom versions with outlier samples with respect to their faithfulness scores removed.

    # Dataset with all samples that have model metric m(M) > 4, used in our main experiments.
    VERB_AGREEMENT_TEST_CONFIDENT_MODEL = ('rc_test_confident_model.json', DatasetCategory.PLAIN_PROMPT)
    # Additional dataset which is a subset of the above, but with additional constraint that
    # they satisfy a circuit metric m(C) > 2 for 1K-sized circuit
    VERB_AGREEMENT_TEST_CONFIDENT_MODEL_SALIENT_CIRCUIT = \
        ('rc_test_confident_model_salient_circuit.json', DatasetCategory.PLAIN_PROMPT)

    # Finally, there are modified versions of the original verb-agreement datasets where 
    # we replaced regular verbs with be-conjugations.
    VERB_AGREEMENT_BE = ('rc_train_processed.csv', DatasetCategory.AGREEMENT_BE)
    VERB_AGREEMENT_TEST_BE = ('rc_test_processed.csv', DatasetCategory.AGREEMENT_BE)

    # --- Other datasets ---
    # HuggingFace datasets
    COMMONSENSE_QA = ("tau/commonsense_qa", DatasetCategory.QA) # HuggingFace dataset
    COMMONSENSE_QA_FILTERED = ("drsis/deception-commonsense_qa_wo_chat", DatasetCategory.QA) # HuggingFace dataset

    # Local true/false datasets
    CITIES = ('cities_true_false.json', DatasetCategory.TRUE_FALSE)
    COMPANIES = ('companies_true_false.json', DatasetCategory.TRUE_FALSE)
    FACTS = ('facts_true_false.json', DatasetCategory.TRUE_FALSE)
    
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