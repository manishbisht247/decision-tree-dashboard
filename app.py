import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import data.data_loader as dd
from model.decision_tree import train_decision_tree
from utils.metrics import calculate_matrics
from utils.plots import plot_confusion_matrix, accuracy_vs_depth


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Decision Tree Explorer",
    layout="wide"
)

st.title("Decision Tree Model Explorer")
st.write("Interactively analyze how Decision Tree parameters affect model performance.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model Parameters")

max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
criterion = st.sidebar.radio("Criterion", ["gini", "entropy"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# -----------------------------
# Load data
# -----------------------------
X, y, _, _ = dd.data_loader()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = train_decision_tree(
    X_train,
    y_train,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion
)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
metrics = calculate_matrics(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred
)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Train Accuracy", f"{metrics['train_accuracy']:.3f}")
col2.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
col3.metric("Overfit Gap", f"{metrics['overfit_gap']:.3f}")

# -----------------------------
# Plots
# -----------------------------
st.subheader("Visual Analysis")

col_left, col_right = st.columns(2)

with col_left:
    fig_cm = plot_confusion_matrix(
        y_test,
        y_test_pred,
        title="Test Confusion Matrix"
    )
    st.pyplot(fig_cm)

with col_right:
    fig_acc = accuracy_vs_depth(X, y)
    st.pyplot(fig_acc)
