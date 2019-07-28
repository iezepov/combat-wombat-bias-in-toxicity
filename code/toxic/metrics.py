import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

POWER = -5
TOXICITY_COLUMN = "target"
IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]


def _power_mean(series, p):
    return np.power(np.power(series, p).mean(), 1 / p)


def compute_toxic_metrics(dataset, model_name):
    """
        Returns tuple of (float, flaot, df), where values are
        (final_metric, overall_auc, bias_metrics_df)
    """

    def compute_masked_auc(mask):
        return roc_auc_score(
            dataset.loc[mask, TOXICITY_COLUMN], dataset.loc[mask, model_name]
        )

    records = []
    for subgroup in IDENTITY_COLUMNS:
        sub_col = dataset[subgroup]
        xor_col = sub_col ^ dataset[TOXICITY_COLUMN]
        record = {
            "subgroup": subgroup,
            "subgroup_size": sub_col.sum(),
            "subgroup_auc": compute_masked_auc(sub_col),
            "bpsn_auc": compute_masked_auc(xor_col),
            "bnsp_auc": compute_masked_auc(~xor_col),
        }
        records.append(record)

    bias_df = pd.DataFrame(records)

    bias_score = (
        _power_mean(bias_df["subgroup_auc"], POWER)
        + _power_mean(bias_df["bpsn_auc"], POWER)
        + _power_mean(bias_df["bnsp_auc"], POWER)
    ) / 3

    overall_auc = roc_auc_score(dataset[TOXICITY_COLUMN], dataset[model_name])

    final_metric = 0.25 * overall_auc + 0.75 * bias_score

    return final_metric, overall_auc, bias_df
