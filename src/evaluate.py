import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix

def get_continuous_metrics(y_true, y_pred):
    """Calculates RMSE and MAE for the raw temperature regression."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}

def get_classification_metrics(y_true, y_pred):
    """Calculates classification metrics including the Peirce skill score."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Peirce Skill Score (True Positive Rate - False Positive Rate)
    # labels=[0, 1] is required, not optional: without it, confusion_matrix
    # returns a 1x1 matrix (and .ravel() crashes) whenever a batch happens
    # to contain only one class -- e.g. a small fold with zero frost days.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    peirce_skill = tpr - fpr
    
    return {"precision": precision, "recall": recall, "f1": f1, "peirce_skill": peirce_skill}