# Decision Tree Model Explorer (Streamlit Dashboard)

This project is an interactive **Streamlit-based dashboard** designed to help users understand how **Decision Tree model parameters** affect performance, overfitting, and generalization.

Instead of treating Decision Trees as a black box, this app allows users to **experiment visually** with key hyperparameters and immediately observe their impact on metrics and plots.

---

## 🎯 What This Project Does

- Trains a **Decision Tree Classifier** on a real-world dataset
- Allows users to tune important hyperparameters interactively
- Displays **train vs test performance**
- Visualizes **overfitting behavior**
- Helps build intuition about model complexity

This project focuses on **learning and explainability**, not just accuracy.

---

## 🧠 Concepts Demonstrated

- Decision Tree hyperparameters:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `criterion (gini / entropy)`
- Overfitting vs underfitting
- Train vs test performance comparison
- Confusion Matrix interpretation
- Model complexity analysis

---

## 🧪 Dataset Used

- **Breast Cancer Wisconsin Dataset**
- Binary classification (benign vs malignant)
- Clean and well-structured, ideal for model behavior analysis

---

## 📊 Features in the Dashboard

### 1. Interactive Controls
Users can adjust:
- Tree depth
- Minimum samples for split and leaf
- Split criterion
- Train-test split ratio

### 2. Performance Metrics
Displayed in real time:
- Train Accuracy
- Test Accuracy
- Overfitting Gap (Train − Test Accuracy)

### 3. Visual Analysis
- **Confusion Matrix** for test predictions
- **Accuracy vs Max Depth** curve to visualize overfitting

---

## 🗂 Project Structure

decision-tree-dashboard/
│
├── app.py # Streamlit application
├── data/
│ └── data_loader.py # Dataset loading logic
├── model/
│ └── decision_tree.py # Model training logic
├── utils/
│ ├── metrics.py # Evaluation metrics
│ └── plots.py # Visualization utilities
├── requirements.txt
└── README.md



Each component is modular and reusable.

---

## ▶️ How to Run the Project

### 1. Create and activate environment

        conda create -n dt_dashboard python=3.10
        conda activate dt_dashboard

### 2. Install dependencies
        pip install -r requirements.txt

### 3. Run the Streamlit app
        streamlit run app.py