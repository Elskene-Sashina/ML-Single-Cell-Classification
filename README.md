# 🧬 ML Single Cell Classification Project

This project aims to build a classifier that predicts brain cell types using single-cell RNA sequencing (scRNAseq) data. We use gene expression data (`counts.h5ad`) and cell annotations (`cell_labels.csv`) to develop models that distinguish between three cell classes: **GABAergic**, **Glutamatergic**, and **Other**.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Environment Setup](#environment-setup)
- [Some important part](#some-important-part)

---

## 🔍 Overview

In this project, we:

- **Load and explore** scRNAseq data and cell annotations using the `scanpy` library  
- **Preprocess** the data by integrating quality control metrics and filtering low-quality cells  
- **Preporation data**  -  split the data into training, validation, and test sets  
- **Train and evaluate** multiple classifiers (KNN, Logistic Regression, Random Forest, and SVM) with hyperparameter tuning using scikit-learn

---

## 📂 Data
 _Note:_ Data can be downloaded from [here](https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/processed_data/).
- **Gene Expression Data:**  
  The file `counts.h5ad` contains a normalized and preprocessed gene expression matrix of **280,186 cells × 254 genes**.  
- **Cell Annotations:**  
  The file `cell_labels.csv` includes cell type labels and metadata.  


---

## 🛠️ Environment Setup


Install and import the required packages. For example, install `scanpy` with:

```bash
!pip install scanpy
```

Then, import the necessary libraries:
```python
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import pooch  # For data retrieval
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
```
---

## 🎉 Some important part

Make shure that you loaded and changed path in **2.1. Load Annotation and Expression Data**

cell_labels = pd.read_csv("/YOUR/PATH/HERE/cell_labels.csv", index_col=0)
adata = sc.read_h5ad("/YOUR/PATH/HERE/counts.h5ad")

Happy modeling! 🚀
