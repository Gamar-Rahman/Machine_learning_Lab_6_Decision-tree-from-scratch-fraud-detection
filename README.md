# Machine_learning_Lab_6_Decision-tree-from-scratch-fraud-detection
From-scratch Decision Tree using Gini index to classify authentic vs counterfeit banknotes—explainable ML aligned with fraud/security detection.
# Decision Tree From Scratch — Banknote Authentication (Cybersecurity + Cloud Security)

This project implements a **binary decision tree classifier from scratch** (no sklearn decision tree) using the **Gini index** and **greedy recursive splitting** to classify banknotes as authentic vs counterfeit.

In security terms, this is a practical baseline for **fraud detection** and **anomaly/risk scoring** pipelines that are common in cloud environments (e.g., API abuse detection, account takeover risk scoring, transaction fraud signals).

---

## Why this matters for Cybersecurity
Counterfeit detection is a real-world analogue to security classification problems:

- **Fraud detection** (payments, identity, counterfeit goods)
- **Risk scoring** for suspicious activity (behavioral signals → malicious/benign)
- **Explainable detection**: decision trees are interpretable and help analysts understand *why* a prediction happened

Decision trees are widely used in operational security because they provide:
✅ explainability  
✅ fast inference  
✅ strong performance with simple features

---

## Dataset
**Banknote Authentication Dataset**
- 1,372 samples
- 4 numeric features (derived from banknote image statistics)
- binary label (0/1)

---

## What I Implemented (From Scratch)
### Core building blocks
- gini_score(groups, classes)
  Computes split quality (lower is better).

- create_split(index, threshold, datalist)
  Splits data into left/right groups based on a feature threshold.

- get_best_split(datalist)  
  Tries **every feature** and **every candidate threshold** in the data to find the split that minimizes Gini.

### Tree construction
- to_terminal(group)
  Creates a terminal node by returning the majority class.

- recursive_split(node, max_depth, min_size, depth)`  
  Builds the tree recursively with stopping conditions:
  - empty left/right group
  - maximum depth reached
  - group size <= min_size

- build_tree(train, max_depth, min_size)  
  Creates root split then expands the tree.

### Prediction
- predict(root, sample)
  Traverses the tree (iterative navigation) until it reaches a terminal node.

---

## Training Setup
- First 1000 samples: training
- Remaining samples: testing
- Hyperparameters:
  - `max_depth = 6`
  - `min_size = 10`

Outputs:
- Test Accuracy
- Test F1 Score

