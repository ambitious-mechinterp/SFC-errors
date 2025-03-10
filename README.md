# tim-taras-sfc-errors

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

AISC project aimed to understand the functional role of error nodes in SFC

# Stage-1 analysis
## Motivation and Approach
When examining a high importance scores of an error nodes in Sparse Feature Circuits (SFCs), we need to understand what makes them significant. Mathematically, they are computed as Attribution patching (AtP) scores as dot product between:

- The gradient of the layer's output (resid/mlp/attention) with respect to the metric
- The activation difference between clean and patched prompts for that error node

We wanted to explore whether there are error nodes where:

- The activation difference and gradient are not well-aligned
- Cases with high gradients but low activation differences (or vice versa)

Since these situations would result in low AtP scores, they might be missed in threshold-based filtering.

## Operationalization
From our Objectives page, the task was operationalized in two experiments:
- **Task A:** Identify factors driving high error node scores (AtP/IG)
- **Task B:** Evaluate alignment between gradient-based error node approximations and activation patching-based approximations

## Implementation
There are 3 main notebooks added to solve these tasks:
1. `compute_error_scores.ipynb`: computes SFC error scores using attribution patching and saves them for future analysis
2. `analyze_error_scores.ipynb`: performs **Task A**, answering questions from the _Motivation and Approach_
3. `validate_error_scores.ipynb`: performs **Task B**, studying the correlation between SFC error scores and corresponding activation patching scores

Task A and B are logically independent, but both depend on the execution of `compute_error_scores.ipynb` notebook. So, if you want to reproduce the results, run the notebooks in the following order:
- `compute_error_scores.ipynb` ->  `analyze_error_scores.ipynb`
- `compute_error_scores.ipynb` ->  `validate_error_scores.ipynb`

## For code reviewers

All 3 notebooks follow the same template (reflected in the heading namings):

---
### **Shared Part**

- Setup
- Loading the model
- Loading the data
- Setting up the SAEs

---

### **Notebook-Specific Part**

- Either **SFC scores computation**, OR **error metrics computation**, OR **act patching effects for error computation**

---

### **Result Analysis Part**

- Load the data computed in the **Notebook-Specific Part** and analyze it using plots

---

- To understand the implementation, only checking the **Notebook-Specific Part** is needed. Usually, it just calls a single method from my `SFC_Gemma` class.
- To understand the results, only checking the **Result Analysis Part** is needed.

So even though this PR says **+37,957 lines added**, most of them (I think) come from:

1. Supporting code for loading the dataset in an SFC-friendly format
2. SFC computation itself (`SFC_Gemma` class)
3. A lot of duplicate setup code from the **Shared Part** across notebooks

Thus, I’d recommend just checking the crucial pieces of code from the **Notebook-Specific Part** and **Result Analysis Part**.  
And feel free to ignore the plotting code unless you want to go crazy! Anyway, this was primarily Claude’s part. 😆

## Methodology and results
### Task A
In this task, we computed the following data:

- Activation Difference Norms - `activation_diff_norm`: How much the error node activations change (in magnitude) between clean and patched inputs (patched - clean)
- Gradient Norms - `gradient_norm`: The magnitude of the gradient for each error node
- Attribution patching Scores - `atp_score`: The original attribution patching scores (dot product of gradient and activation difference)

Then we implemented scatter plots visualizing these data:

- Activation difference norms vs. gradient norms (colored by AtP score)
- Activation difference norms vs. cosine similarity between activation difference and gradient vectors (colored by AtP score)

### Results summary
![Activation difference norms vs. gradient norms (colored by AtP score)](https://raw.githubusercontent.com/ambitious-mechinterp/SFC-errors/26f90ddb61f152c582506a78430d41ee65b3a335/reports/figures/error_metrics.png?token=ARM4RWIHSEHMCZL5TB5NWG3HZG4WC)

#### **Resid error nodes**:
- Activation difference norm is perfectly correlated with the layer number: the greater is the layer of the error node, the more it differs (in norm) between the clean and patched settings
- Activation difference has an interesting non-linear relationship with the AtP scores: **the AtP scores are the greatest when the Activation difference norm is in its middle values** (not too high & not too low)

#### **MLP error nodes**:
- Activation difference is still perfectly correlated with the layer number, but **it's not longer** that predictive of the error score

#### **Attention error nodes**:
- Here the plots looked basically random... the only takeaway we have is that the gradients here are much higher in norm than in the previous cases, and the activation difference on the other hand is much lower. But neither is predictive of anything.

#### Does the alignment with the gradient even matter?
Yes, as our `Activation difference norms vs. cosine similarity between activation difference and gradient vectors` plot showed, alignment of activation difference and gradient vector does matter (because otherwise the late error node would have a large AtP score, but they don't)

#### Resid activations
**Question**: will we see the same patterns if we take resid_post activations instead of the error nodes?
- The gradients should be the same (mathematically they are)
- But norm(activation_diff) is not guaranteed to in the same way

**Answer**: 
- residual activation difference grew much faster in norm than in the error nodes case
- it no longer showed the non-linear pattern: here the AtP scores grow monotically with norm(activation_diff)

![Residual activations: activation difference norms vs. gradient norms (colored by AtP score)](https://raw.githubusercontent.com/ambitious-mechinterp/SFC-errors/26f90ddb61f152c582506a78430d41ee65b3a335/reports/figures/resid_metrics.png?token=ARM4RWOQH2I5JCSJFZ5CTKTHZG4WC)

### Task B
Here the results are less lengthy, as the task was simply to compute the correlation between the Error nodes SFC scores and Activation-patching scores. The figure below shows that it's indeed the case, confirming the results from the SFC paper.

![SFC error scores vs act patching scores](https://raw.githubusercontent.com/ambitious-mechinterp/SFC-errors/26f90ddb61f152c582506a78430d41ee65b3a335/reports/figures/error_patching.png?token=ARM4RWLQAFWWSOPW3Q57SPTHZG4WC)

The same holds for MLP and Attention error nodes as you can see in the `validate_error_scores.ipynb` notebook.
