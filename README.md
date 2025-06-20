# Investigating the Role of Error Nodes in Sparse Feature Circuits

This repository contains the code and experiments for the blog post "[What is the functional role of SAE Errors?](placeholder_for_blog_post_link)". Our work investigates the role of SAE error terms within Sparse Feature Circuits (SFCs), focusing on the Gemma-2 model and Gemma Scope SAEs, as detailed in [Marks et al., 2025](https://arxiv.org/abs/2403.19647).

The core of our investigation is a case study on a subject-verb agreement task, where we explore the hypothesis that error nodes may represent intermediate computational steps of features formed via cross-layer superposition.

## Setup

To set up the environment and install the required dependencies, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate the Conda environment:**
    This project uses Conda for environment management. You can create the environment from the provided file:
    ```bash
    conda env create -f environment.yml
    conda activate sfc-errors
    ```

3.  **Install the project package:**
    To make the local modules available for import in the notebooks, install the project in editable mode:
    ```bash
    pip install -e .
    ```

## Data

The datasets required to run the experiments are located in the `data/` directory. The primary dataset used for the main experiments is `rc_test_confident_model.json`. This is a filtered version of the subject-verb agreement task dataset, containing only prompts where the model exhibits high confidence in its predictions, leading to more stable faithfulness metrics with no outliers.

## Codebase Structure

The codebase is organized into a core library under `sfc-errors/` and a set of Jupyter notebooks in `notebooks/`.

### Core Library (`sfc-errors/`)

The core logic is divided into three main components:

*   **Dataset Processing**:
    *   `sfc-errors/classes/sfc_data_loader.py`: A versatile data loader responsible for loading, processing, and preparing datasets for experiments. It handles tokenization, padding, applying chat templates, and generating the clean/patched data pairs required for SFC analysis.
    *   `sfc-errors/utils/enums.py`: Defines several enumerations (`SupportedDatasets`, `DatasetCategory`) to standardize dataset handling and processing pipelines.
    *   `sfc-errors/utils/prompts.py`: (Not used for the post) A collection of system and task prompts used throughout our previous experiments, not related to the post.

*   **SFC Score Computation**:
    *   `sfc-errors/classes/sfc_model.py`: A wrapper class for the Gemma-2 model that automates loading and attaching Gemma Scope SAEs. It contains the primary implementation for computing attribution patching (AtP) scores for both SAE features and error nodes. Currently doesn't support Integrated Gradients (IG) or edge weight calculation.
    *   `sfc-errors/classes/sfc_node_scores.py`: A helper class for managing computed node scores. It encapsulates logic for initializing, aggregating, selecting, and persisting SFC scores (our general term for AtP/IG scores, although only AtP is supported currently).

*   **Circuit Evaluation**:
    *   `sfc-errors/classes/sfc_evaluator.py`: The main engine for the ablation and restoration experiments. It implements the faithfulness metric and provides a unified interface to evaluate circuits by ablating and/or restoring any combination of nodes based on various criteria (e.g., score thresholds, node names, position indices stc.).

### Notebooks (`notebooks/`)

The `notebooks/` directory contains the workflows for reproducing the results presented in the blog post. It is split into two main subdirectories, `error_scores/` and `main/`, which correspond to the two major stages of the investigation.

## Reproducing the Blog Post Results

The following sections detail the steps to reproduce the key findings and figures from our blog post.

### Part 1: Error Score Analysis

**Goal:** To analyze the components of the error node AtP scores (activation difference and gradient) and to validate that AtP scores are a reliable proxy for the true activation patching effect.

**Location:** `notebooks/error_scores/`

**Execution Order:**

1.  **`compute_error_scores.ipynb`**: This is the starting point. It runs the model on the subject-verb agreement task to compute and save the necessary data for analysis:
    *   Attribution Patching (AtP) scores
    *   Activation difference norms
    *   Gradient norms
    These artifacts are saved to disk and used by the subsequent notebooks.

- **`analyze_error_scores.ipynb`**: This notebook analyzes the individual components of the AtP score (the gradient and the activation difference). This analysis was part of our initial exploration to understand the factors driving error node scores, but it was not included in the final blog post.

- **`validate_error_scores.ipynb`**: This computes the true activation patching effect for error nodes and plots its correlation against the AtP scores, confirming that AtP is a valid approximation for our use case.

[`analyze_error_scores.ipynb` and `validate_error_scores.ipynb` are independent and can be run in any order, **assuming `compute_error_scores.ipynb` has been run**.]

### Part 2: Main Ablation & Restoration Experiments

**Goal:** To test the hypothesis that error nodes serve as intermediate representations for SAE features formed via cross-layer superposition. This is done through a series of ablation and restoration experiments.

**Location:** `notebooks/main/`

**Execution Order:**

1.  **`compute_mean_activations.ipynb`**: A necessary preliminary step. This notebook calculates the mean activation value for every SAE feature and error node across the dataset. These values are used for mean-ablation during the faithfulness evaluations.

2.  **`analyze_faithfulness_outliers.ipynb`**: This notebook analyzes the distribution of the faithfulness metric and is used to create the filtered dataset (`rc_test_confident_model.json`) by removing prompts where the model is not confident in the right verb form.

3.  **`evaluate_circuit_faithfulness.ipynb`**: This notebook replicates the circuit faithfulness evaluation from the original SFC paper (Figure 3). It serves as a validation of our `CircuitEvaluator` implementation.

4.  **`ablation_restoration_main.ipynb`**: This is the central notebook for this project. It contains the code for all the main ablation and restoration experiments described in the blog post, including:
    *   Ablating the error nodes and restoring top-K SAE features.
    *   Ablating the error nodes up to a specific threshold and restoring top-25 features
    *   The "sliding window" and "expanding window" ablation experiments from the Targeted error ablation section.

This notebook makes extensive use of the `CircuitEvaluator`. `CircuitEvaluator.evaluate_circuit_faithfulness` is a core method in our experimental framework. It has grown quite complex and currently operates in two main modes, which ideally would be refactored into separate functions for clarity. If you plan to use this for your own research, feel free to reach out, and we can prioritize refactoring it.

The method's primary function is to evaluate the faithfulness of a given circuit by selectively ablating some nodes while restoring others. Its behavior is primarily controlled by the following arguments:
- **Mode 1: Threshold-based Circuit Evaluation** (`node_threshold` is provided): In this mode, the circuit is defined by including all nodes with an SFC score *above* the specified `node_threshold`. Everything else is ablated. This is used in `evaluate_circuit_faithfulness.ipynb` to replicate the original SFC paper's faithfulness curve.
- **Mode 2: Custom Ablation/Restoration** (`node_threshold` is `None`): This mode allows for fine-grained, custom interventions. It relies on arguments like `nodes_to_always_ablate`, `nodes_to_restore`, `feature_indices_to_ablate`, and `feature_indices_to_restore` to specify exactly which nodes (or even which specific features within an SAE node) should be ablated or have their original activations patched back in. This flexible mode is used for all the targeted experiments in `ablation_restoration_main.ipynb`.
