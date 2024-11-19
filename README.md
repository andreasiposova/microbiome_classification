# Colorectal Cancer Prediction Using Gut Microbiome Data

## Project Overview
This project aims to classify colorectal cancer (CRC) using supervised machine learning (ML) algorithms applied to gut microbiome data. It evaluates the effectiveness of these algorithms in distinguishing between healthy and CRC samples and explores how preprocessing techniques, such as feature abundance limits, influence classification performance.

## Motivation
CRC is a global health concern where early detection can significantly improve outcomes. Current diagnostic methods are invasive and costly, but leveraging gut microbiome data provides a non-invasive alternative. By using ML techniques, this project seeks to optimize the predictive performance and evaluate preprocessing steps to enhance detection accuracy.

---

## Data and Preprocessing

### Dataset
The dataset originates from Yang et al.’s study, including microbiome data for two patient cohorts:
- **Fudan Cohort**: Used for training.
- **Huadong Cohort**: Used as an external validation set.

### Features
- **Microbiome Data**: Relative abundances of taxonomic units such as genus and family.
- **Biodiversity Metrics**: Shannon, Simpson, and other diversity indices.
- **Transformation**: Discretization of feature values into bins using feature abundance limits (low, medium, high levels).

### Preprocessing Steps
1. **Normalization**: Min-max scaling to standardize feature values.
2. **Handling Infinite Values**: Replacing with maximum observed values in respective columns.
3. **Feature Selection**:
   - Taxonomic levels: Focused on genus and family.
   - Reduced feature space for computational efficiency.

---

## Machine Learning Pipeline

### Algorithms
- **Random Forest (RF)**
- **XGBoost (XGB)**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**

### Coding Steps
1. **Preprocessing**:
   - Cleaning and normalizing microbiome data.
   - Discretizing features with abundance limits using a binning function.
2. **Feature Selection**:
   - Using predefined lists or recursive feature elimination to match original paper’s selection.
3. **Model Training**:
   - Implementing ML models using `scikit-learn` and `xgboost`.
   - Hyperparameter tuning with `GridSearchCV` to optimize performance.
4. **Cross-Validation**:
   - Splitting data with 10-fold cross-validation.
   - Training and evaluating models to avoid overfitting.
5. **Evaluation Metrics**:
   - Calculating ROC AUC, accuracy, precision, recall, F1, and F2 scores using `sklearn.metrics`.
   - Generating confusion matrices for detailed analysis.

---

## Experiment Configurations
Three main dataset configurations were tested:
1. **Young-Onset vs. Healthy**
2. **Old-Onset vs. Healthy**
3. **Combined Dataset (No Age Group Distinction)**

For each configuration:
- Models were trained and validated with and without feature selection.
- Discretization of feature values was applied at different abundance levels.

---

## Key Results

### Old-Onset Samples
- **Random Forest** achieved the highest performance with selected features (ROC AUC: 81.07%).
- Discretization improved recall, reducing FN predictions.

### Young-Onset Samples
- **Random Forest** delivered the best results (ROC AUC: 69.38%) with feature selection.
- Discretization at the low abundance level significantly improved recall, especially for KNN and SVM.

### Combined Dataset
- Combining young- and old-onset samples improved classification results:
  - **Random Forest**: ROC AUC of 77.85%.
  - Discretization at the low level enhanced recall and reduced FNs for all models.

---

## Key Observations
1. **Impact of Feature Selection**:
   - Improves computational efficiency and interpretability.
   - Slight performance gains for most algorithms.
2. **Discretization Benefits**:
   - Reduces FN rates, crucial for medical diagnostics.
   - Particularly impactful for distance-based models (KNN and SVM).
3. **Algorithm Performance**:
   - Random Forest consistently outperformed other models.
   - XGBoost showed comparable results with optimized hyperparameters.

---

## References
[1] D. Fern´andez-Edreira, J. Li˜nares-Blanco, and C. Fernandez-Lozano. Machine learning analysis
of the human infant gut microbiome identifies influential species in type 1 diabetes. Expert
Systems with Applications, 185:115648, 2021.
[2] M. Hittmeir, R. Mayer, and A. Ekelhart. Distance-based techniques for personal microbiome
identification. In Proceedings of the 17th International Conference on Availability, Reliability
and Security, pages 1–13, 2022.
[3] U. Ladabaum, J. A. Dominitz, C. Kahi, and R. E. Schoen. Strategies for colorectal cancer
screening. Gastroenterology, 158(2):418–432, 2020.
[4] B. A. Peters, M. Wilson, U. Moran, A. Pavlick, A. Izsak, T. Wechter, J. S. Weber, I. Osman,
and J. Ahn. Relating the gut metagenome and metatranscriptome to immunotherapy responses
in melanoma patients. Genome medicine, 11(1):1–14, 2019.
[5] B. D. Top¸cuo˘glu, N. A. Lesniak, M. T. Ruffin IV, J. Wiens, and P. D. Schloss. A framework for
effective application of machine learning to microbiome-based classification problems. MBio,
11(3):e00434–20, 2020.
[6] B. D. Top¸cuo˘glu, N. A. Lesniak, M. T. Ruffin, J. Wiens, and P. D. Schloss. A framework for
effective application of machine learning to microbiome-based classification problems. mBio,
11(3):10.1128/mbio.00434–20, 2020.
[7] Y. Yang, L. Du, D. Shi, C. Kong, J. Liu, G. Liu, X. Li, and Y. Ma. Dysbiosis of human gut
microbiome in young-onset colorectal cancer. Nature communications, 12(1):1–13, 2021.
