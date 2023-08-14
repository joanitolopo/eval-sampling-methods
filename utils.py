import os
from sklearn import metrics
from sklearn.metrics import f1_score
import math
from tabulate import tabulate

def get_models_folder_path(folder_name):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    models_folder_path = os.path.join(current_folder, folder_name)
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
    return models_folder_path


def plot(model, X_train, y_train, X_testdata, y_testdata):
    p_train = model.predict(X_train)
    p_testdata = model.predict(X_testdata)

    roc_auc_testdata = metrics.roc_auc_score(y_testdata, p_testdata) * 100
    f1_testdata = f1_score(y_testdata, p_testdata, average='macro') * 100
    
    precision = metrics.precision_score(y_testdata, p_testdata) * 100
    recall_sensitivity = metrics.recall_score(y_testdata, p_testdata, pos_label=1) * 100
    recall_specificity = metrics.recall_score(y_testdata, p_testdata, pos_label=0) * 100

    g_mean = math.sqrt(recall_sensitivity * recall_specificity)

    metrics_score = [
    ("ROC AUC", roc_auc_testdata),
    ("F1 Score", f1_testdata),
    ("Precision", precision),
    ("Recall (Sensitivity)", recall_sensitivity),
    ("Recall (Specificity)", recall_specificity),
    ("G-Mean", g_mean)]

    metrics_table = tabulate(metrics_score, headers=["Metric", "Value"], tablefmt="plain")

    return metrics_table
    
