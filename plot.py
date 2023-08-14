from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import math
from logger import logging
from tabulate import tabulate

def plot(model, X_train, y_train, X_testdata, y_testdata):
    p_train = model.predict(X_train)
    p_testdata = model.predict(X_testdata)

    roc_auc_testdata = metrics.roc_auc_score(y_testdata, p_testdata) * 100
    f1_train = f1_score(y_train, p_train, average='macro') * 100
    f1_testdata = f1_score(y_testdata, p_testdata, average='macro') * 100
    
    precision = metrics.precision_score(y_testdata, p_testdata) * 100
    recall_sensitivity = metrics.recall_score(y_testdata, p_testdata, pos_label=1) * 100
    recall_specificity = metrics.recall_score(y_testdata, p_testdata, pos_label=0) * 100

    g_mean = math.sqrt(recall_sensitivity * recall_specificity)

    metrics = [
    ("ROC AUC", roc_auc_testdata),
    ("F1 Score", f1_testdata),
    ("Precision", precision),
    ("Recall (Sensitivity)", recall_sensitivity),
    ("Recall (Specificity)", recall_specificity),
    ("G-Mean", g_mean)]

    metrics_table = tabulate(metrics, headers=["Metric", "Value"], tablefmt="plain")

    logging.info("\nMetrics Summary:\n" + metrics_table)



