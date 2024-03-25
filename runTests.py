import httpx
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import json
import asyncpg
import os
import asyncio
import datetime

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StructuredDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from aif360.explainers import MetricTextExplainer
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.detectors.mdss.ScoringFunctions import Bernoulli
from aif360.detectors.mdss.MDSS import MDSS
from aif360.detectors.mdss.generator import get_random_subset

orig_dataset = pd.read_csv('usersWatch.csv')
user_watch_dataset = orig_dataset.sample(n=100000)
features_4_scanning = ['gender', 'age', 'occupation']

#print(user_watch_dataset.describe())
# Filter the dataframe to contain only rows with 'high' in the 'watchtime' column
specific_value_df = user_watch_dataset[user_watch_dataset['watchtime'] >= 45]

# Randomly sample 3 rows from the dataframe with the specific value
specific_value_samples = specific_value_df.sample(n=3)

# Randomly sample 2 rows from the original dataframe (excluding the ones with the specific value)
other_samples = user_watch_dataset.drop(specific_value_samples.index).sample(n=2)

# Concatenate both sets of samples
random_samples = pd.concat([specific_value_samples, other_samples])

print(random_samples)

def print_report(data, subset):
    """ Utility function to pretty-print the subsets."""
    if subset:
        to_choose = user_watch_dataset[subset.keys()].isin(subset).all(axis = 1)
        df = user_watch_dataset[['watchtime_YN', 'predicted_watch']][to_choose]
    else:
        for col in features_4_scanning:
            subset[col] = list(user_watch_dataset[col].unique())
        df = user_watch_dataset[['watchtime_YN', 'predicted_watch']]

    true = df['watchtime_YN'].sum()
    pred = df['predicted_watch'].sum()
    
    print('\033[1mSubset: \033[0m')
    print(subset)
    print('\033[1mSubset Size: \033[0m', len(df))
    print('\033[1mTrue Watch: \033[0m', true)
    print('\033[1mPredicted Watch: \033[0m', pred)
    print()
    
np.random.seed(11)
random_subset = get_random_subset(user_watch_dataset[features_4_scanning], prob = 0.05, min_elements = 30000)
#print_report(user_watch_dataset, random_subset)

# Bias scan
scoring_function = Bernoulli(direction='negative')
scanner = MDSS(scoring_function)
 
scanned_subset, _ = scanner.scan(user_watch_dataset[features_4_scanning], 
                        expectations = user_watch_dataset['predicted_watch'],
                        outcomes = user_watch_dataset['watchtime_YN'], 
                        penalty = 1, 
                        num_iters = 1,
                        verbose = False)

#print_report(user_watch_dataset, scanned_subset)
print_report(user_watch_dataset, {'gender':[1]})
print_report(user_watch_dataset, {'gender':[0]})

def convert_to_standard_dataset(df, target_label_name, scores_name=""):

    # List of names corresponding to protected attribute columns in the dataset.
    # Note that the terminology "protected attribute" used in AI Fairness 360 to
    # divide the dataset into multiple groups for measuring and mitigating 
    # group-level bias.
    protected_attributes=['gender']
    
    # columns from the dataset that we want to select for this Bias study
    selected_features = [ 'age', 'occupation', 'gender', 'predicted_probability']
    
    # This privileged class is selected based on MDSS subgroup evaluation.
    # in previous steps. In our case non-homeowner (homeowner=0) are considered to 
    # be privileged and homeowners (homeowner=1) are considered as unprivileged.
    privileged_classes = [[1]]   

    # Label values which are considered favorable are listed. All others are 
    # unfavorable. Label values are mapped to 1 (favorable) and 0 (unfavorable) 
    # if they are not already binary and numerical.
    favorable_target_label = [1]

    # List of column names in the DataFrame which are to be expanded into one-hot vectors.
    categorical_features = ['age', 'occupation']

    # create the `StandardDataset` object
    standard_dataset = StandardDataset(df=df, label_name=target_label_name,
                                    favorable_classes=favorable_target_label,
                                    scores_name=scores_name,
                                    protected_attribute_names=protected_attributes,
                                    privileged_classes=privileged_classes,
                                    categorical_features=categorical_features,
                                    features_to_keep=selected_features)
    if scores_name=="":
        standard_dataset.scores = standard_dataset.labels.copy()
        
    return standard_dataset

# Create two StandardDataset objects - one with true conversions and one with
# predicted conversions.

# First create the predicted dataset
user_watch_dataset_pred = convert_to_standard_dataset(user_watch_dataset, 
                                            target_label_name = 'predicted_watch',
                                            scores_name = 'predicted_probability')

# Use this to create the original dataset
user_watch_dataset_orig = user_watch_dataset_pred.copy()
user_watch_dataset_orig.labels = user_watch_dataset["watchtime_YN"].values.reshape(-1, 1)
user_watch_dataset_orig.scores = user_watch_dataset["watchtime_YN"].values.reshape(-1, 1)

# After converting dataset to Standard dataset your privileged class will always be 1 
# & the others would be 0 . If the column is already binary it doesn't convert to 0 & 1.

privileged_groups= [{'gender': 1}]
unprivileged_groups = [{'gender': 0}]

metric_orig = BinaryLabelDatasetMetric(user_watch_dataset_orig, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print(f"Disparate impact for the original dataset = {metric_orig.disparate_impact():.4f}")
print()
print()

metric_pred = BinaryLabelDatasetMetric(user_watch_dataset_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print(f"Disparate impact for the predicted dataset = {metric_pred.disparate_impact():.4f}")
print()
print()
# Metrics function
from collections import OrderedDict
from aif360.metrics import ClassificationMetric

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics


# Split the standard dataset into train, test and validation 
# (use the same random seed to ensure instances are split in the same way)
random_seed = 1001
dataset_orig_train, dataset_orig_vt = user_watch_dataset_orig.split([0.7], 
                                                shuffle=True, seed=random_seed)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], 
                                                shuffle=True, seed=random_seed+1)

print(f"Original training dataset shape: {dataset_orig_train.features.shape}")
print(f"Original validation dataset shape: {dataset_orig_valid.features.shape}")
print(f"Original testing dataset shape: {dataset_orig_test.features.shape}")
print()
dataset_pred_train, dataset_pred_vt = user_watch_dataset_pred.split([0.7], 
                                                shuffle=True, seed=random_seed)
dataset_pred_valid, dataset_pred_test = dataset_pred_vt.split([0.5], 
                                                shuffle=True, seed=random_seed+1)

print(f"Predicted training shape: {dataset_pred_train.features.shape}")
print(f"Predicted validation shape: {dataset_pred_valid.features.shape}")
print(f"Predicted testing shape: {dataset_pred_test.features.shape}")
print()
# Best threshold for classification only (no fairness)

num_thresh = 300
ba_arr = np.zeros(num_thresh)
class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
for idx, class_thresh in enumerate(class_thresh_arr):
    
    fav_inds = dataset_pred_valid.scores > class_thresh
    dataset_pred_valid.labels[fav_inds] = dataset_pred_valid.favorable_label
    dataset_pred_valid.labels[~fav_inds] = dataset_pred_valid.unfavorable_label
    
    classified_metric_valid = ClassificationMetric(dataset_orig_valid,
                                             dataset_pred_valid, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    ba_arr[idx] = 0.5*(classified_metric_valid.true_positive_rate()\
                       +classified_metric_valid.true_negative_rate())

best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
best_class_thresh = class_thresh_arr[best_ind]

print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)
print()

# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05
        
#random seed for calibrated equal odds prediction
np.random.seed(1)

# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]
if metric_name not in allowed_metrics:
    raise ValueError("Metric name should be one of allowed metrics")

# Fit the method
ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                            low_class_thresh=0.01, high_class_thresh=0.99,
                                            num_class_thresh=100, num_ROC_margin=50,
                                            metric_name=metric_name,
                                            metric_ub=metric_ub, metric_lb=metric_lb)
dataset_transf_pred_valid = ROC.fit_predict(dataset_orig_valid, dataset_pred_valid)

print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
print("Optimal ROC margin = %.4f" % ROC.ROC_margin)
print()
metric_pred_valid_transf = BinaryLabelDatasetMetric(dataset_transf_pred_valid, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Postprocessed predicted validation dataset")
print(f"Disparate impact of unprivileged vs privileged groups = {metric_pred_valid_transf.disparate_impact():.4f}")
print()

# Metrics for the test set
fav_inds = dataset_pred_test.scores > best_class_thresh
dataset_pred_test.labels[fav_inds] = dataset_pred_test.favorable_label
dataset_pred_test.labels[~fav_inds] = dataset_pred_test.unfavorable_label

print("Test set")
print("Raw predictions - No fairness constraints, only maximizing balanced accuracy")

metric_test_bef = compute_metrics(dataset_orig_test, dataset_pred_test, 
                unprivileged_groups, privileged_groups)

# Metrics for the transformed test set
dataset_transf_pred_test = ROC.predict(dataset_pred_test)

print()
print("Test set")
print("Transformed predictions - With fairness constraints")
metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_pred_test, 
                unprivileged_groups, privileged_groups)


