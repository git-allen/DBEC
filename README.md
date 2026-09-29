# DBEC: Density-Based Entropy Clustering

DBEC is a novel clustering algorithm introduced in the paper [New Topic Discovery Using LLM Analysis and Entropy Based Clustering of Short Texts](https://ieeexplore.ieee.org/abstract/document/11416018), presented at the 2025 IEEE International Conference on Data Mining Workshops (ICDMW).

> [!NOTE]
> **Read the Official Whitepaper:** Discover how DBEC abstracts beyond NLP to act as a universal anomaly detection tool across Finance, Healthcare, Computer Vision, and Genomics. Read the full cross-industry breakdown in the official whitepaper here: [**ARTICLE.md**](./ARTICLE.md).

## Overview

Traditional clustering algorithms often struggle with identifying subtle semantic shifts or inconsistencies in text classification, especially in high-dimensional embedding spaces. DBEC adapts the classic DBSCAN algorithm by shifting the focus from the pure spatial density of data points to the **diversity of preassigned labels** within a spatial region (spatial label entropy). 

By computing spatial entropy directly in the sentence embedding space, DBEC effectively isolates dense but label-diverse regions—termed **"impure regions."** 

## Importance of the Algorithm

Text classification models often suffer from overconfidence when faced with novel or ambiguous texts, producing misleadingly high-confidence labels. Evaluating these models to match user perception is challenging, especially with massive or dynamically changing datasets. 

DBEC solves this by surfacing the hidden structures of inconsistency in your text collections. Identifying high-entropy (impure) clusters is highly informative because they typically indicate:
1. **Mislabeled Data:** Documents that are semantically aligned but have been incorrectly labeled by human annotators or automated systems.
2. **Emerging Topics:** Novel concepts or merging themes that are not represented in the current, static taxonomy.
3. **Ambiguous Content:** Items that legitimately span multiple topics, highlighting the need to refine label definitions.

By isolating these specific regions, DBEC allows practitioners to focus expensive downstream analysis (like Large Language Model prompts or manual human review) only on the areas of the dataset that need it most.

## Possible Use Cases

- **Taxonomy Refinement & Evolution:** For dynamic domains like news, social networks, and scientific literature where new topics emerge continuously, DBEC can flag areas where the existing label taxonomy is failing to capture the content.
- **Efficient LLM-Driven Topic Discovery:** Instead of passing an entire massive text corpus to an LLM for topic modeling (which is computationally expensive and prone to context dilution), DBEC can filter the corpus down to only the impure clusters. The LLM can then provide highly targeted semantic interpretations, suggested re-labelings, or emergent topic candidates.
- **Fraud Detection & System Auditing (Finance):** By embedding financial transaction metadata and passing "Legitimate" vs "Fraudulent" tags, DBEC isolates clusters of structurally identical transactions with conflicting tags—instantly flagging a new fraud tactic or a flaw in automated fraud rules.
- **Image Classification (Computer Vision):** Pass in CNN-extracted feature vectors and predicted image classes. DBEC will find dense regions of visual similarity with high label entropy, perfectly isolating the model's visual "edge cases" or areas where human annotators explicitly disagreed.
- **Electronic Health Records & Misdiagnosis (Healthcare):** By embedding patient medical histories and symptom vectors alongside their diagnosed diseases, DBEC can isolate clusters of patients presenting the exact same physiological symptoms who were given vastly different diagnoses, highlighting high-risk misdiagnoses or overlapping syndromes.
- **E-Commerce Product Categorization (Retail):** Embed product descriptions and pass in store departments as labels. DBEC will find products that are identical in description but split across departments (e.g., "Smart Thermostats" split between *Home Goods* and *Electronics*), signaling the taxonomy needs a dedicated "Smart Home" category.
- **Customer Churn & Journey Analysis (Marketing):** Embed sequence vectors of customer website journeys and pass in their final outcomes ("Purchased", "Churned"). DBEC will find users who took the exact same path but had wildly different outcomes, isolating highly specific friction points.
- **Bioinformatics & Genomics:** DBEC can be applied to domain-specific foundation models. By embedding genomic sequences (e.g., via DNABERT) and supplying functional annotations as labels, DBEC reveals structurally similar sequences that have conflicting functional tags, indicating potential annotation errors, pleiotropy, or significant mutations.

## Usage

The DBEC package is built to be familiar to users of Scikit-Learn. It takes your text embeddings and their current labels to find the impure clusters.

```python
import numpy as np
from dbec import DBEC

# Example: 2D Embeddings and their assigned labels
embeddings = np.array([
    [0.1, 0.1], [0.1, 0.15], [0.15, 0.1], # Cluster 1
    [5.0, 5.0], [5.1, 5.1], [5.0, 5.1]    # Cluster 2
])

# Labels have high diversity in the first cluster, low diversity in the second
labels = ['Topic A', 'Topic B', 'Topic C', 'Topic D', 'Topic D', 'Topic D']

# Initialize DBEC
model = DBEC(
    neighbor=2, 
    min_points_entropy=2, 
    min_samples_high=0.5, 
    entropy_threshold=0.5
)

# Fit and predict the cluster assignments for each point
# Points assigned to '-1' or '(-1,)' are not part of an impure cluster.
impure_cluster_labels = model.fit_predict(embeddings, labels)

print("Cluster Labels:", impure_cluster_labels)

# Capture comprehensive run metrics to evaluate confidence
metrics = model.get_run_metrics(embeddings, labels)

print("Number of Impure Clusters:", metrics['num_clusters'])
print("Total Core Points (C):", metrics['number_core'])
print("Points with High Entropy (H):", metrics['number_high'])
```

## Citation

If you use DBEC in your research, please cite the original paper:

```bibtex
@inproceedings{detmer2025new,
  title={New Topic Discovery Using LLM Analysis and Entropy Based Clustering of Short Texts},
  author={Detmer, Allen and Bhatnagar, Raj and Aurisano, Jillian},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={1180--1189},
  year={2025},
  organization={IEEE}
}
```
