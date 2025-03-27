# Credit-Approval-Prediction-ML-Comparison
A machine learning project comparing Logistic Regression, SVM, Random Forest, KNN, and an Artificial Neural Network (ANN) for predicting credit approval outcomes. Includes data preprocessing, model training, evaluation metrics, and performance analysis on a credit dataset.
### Key Features:
- Data preprocessing and feature scaling.
- Implementation of 5 ML algorithms.
- Evaluation using accuracy, confusion matrices, and classification reports.
- Analysis of model performance on an imbalanced dataset.

## Repository Structure
- `credit_approval_analysis.py`: Main Python script for data processing, model training, and evaluation.
- `Credit.csv`: Dataset (not included here; replace with your dataset).
- `README.md`: This documentation.

## Code Overview

### 1. Dataset Preparation
- Loads `Credit.csv` and splits into features (`X`) and target (`y`).
- Splits data into training (75%) and test (25%) sets.
- Applies `StandardScaler` for feature scaling.

### 2. Models Implemented
1. **Logistic Regression**: Baseline model with 81.5% accuracy.
2. **SVM (RBF Kernel)**: Achieved 82.2% accuracy.
3. **Random Forest**: Slightly better precision for class 1 (35% recall).
4. **KNN**: Lower performance (79.6% accuracy) due to class imbalance.
5. **ANN (Keras)**: Two hidden layers with ReLU, output layer with sigmoid (82.3% accuracy).

### 3. Key Results
| Model                | Accuracy | Class 1 Recall | Class 1 F1-Score |
|----------------------|----------|----------------|------------------|
| Logistic Regression  | 81.47%   | 24%            | 36%              |
| SVM                  | 82.20%   | 33%            | 44%              |
| Random Forest        | 82.23%   | 35%            | 46%              |
| KNN                  | 79.63%   | 34%            | 42%              |
| ANN                  | ~82.3%   | Not Reported   | Not Reported     |

**Imbalance Issue**: The dataset is imbalanced (5880 class 0 vs. 1620 class 1 in the test set), leading to poor recall for class 1 across models.

## How to Use
1. **Dataset**: Place `Credit.csv` in the project directory.
2. **Dependencies**: numpy, pandas, scikit-learn, matplotlib, keras, tensorflow
3. 3. Run the script:
```bash
python credit_approval_analysis.py
