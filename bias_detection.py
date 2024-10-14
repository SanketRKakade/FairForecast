from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def detect_bias(X_test, y_test, model, sensitive_attr):
    dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                 df=X_test, label_names=['Direction'],
                                 protected_attribute_names=[sensitive_attr])

    y_pred = model.predict(X_test)

    metric = ClassificationMetric(dataset, privileged_groups=[{sensitive_attr: 1}],
                                  unprivileged_groups=[{sensitive_attr: 0}],
                                  y_pred=y_pred)

    print(f'Disparate Impact: {metric.disparate_impact()}')
    return metric
