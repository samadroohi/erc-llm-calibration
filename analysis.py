
#%%
from datasets import concatenate_datasets
import numpy as np

#%%
from datasets import load_from_disk
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
#%%
datasets = ['meld', 'emowoz', 'emocx']
models = ['Llama-7B', 'Llama-13B', 'Mistral-7B', 'Zephyr-7B']

data_folder = f"/home/samad/Desktop/ACII/data/"

#%%


# %%
#remove none-value instances
def get_accuracy_scores(outputs_path, model_name, dataset_name):
    output_dataset = load_from_disk(f"{outputs_path}{dataset_name}/first/{model_name}")
    modified_dsets = {}
    f_scores ={}
    for split_name in ['train', 'validation', 'test']:
        # delete confidence column from dataset
        if 'confidence' in output_dataset[split_name].features:
            output_dataset[split_name] = output_dataset[split_name].remove_columns('confidence')
        modified_dsets[split_name] = output_dataset[split_name].filter(lambda x: all(value is not None for value in x.values()))
        f_scores[split_name] = f1_score(modified_dsets[split_name]['ground_truth'], modified_dsets[split_name]['prediction'], average='weighted')
    print(f"model: {model_name}    fscore", f_scores)
    return f_scores
# %%
for dataset in datasets:
    if dataset == 'emowoz':
        for model in models:
            acc = get_accuracy_scores(data_folder, model, dataset)
# %%
acc = get_accuracy_scores(data_folder, 'Zephyr-7B', 'emowoz')
# %%

# Expected Calibration Error
def merge_dataset_splits(output_dataset):
    merged_dataset = concatenate_datasets([output_dataset['train'], output_dataset['validation'], output_dataset['test']])
    return merged_dataset
#%%
def calculate_ece(confidences, predictions, true_labels, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for multiclass classification.
    
    Parameters:
    - confidences: array of confidence scores for the predicted class.
    - predictions: array of predicted class labels.
    - true_labels: array of true class labels.
    - n_bins: number of bins to use for calibration error calculation.
    
    Returns:
    - ece: the Expected Calibration Error.
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
         # Check the type of bin_upper
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) == 0:
            continue

        # Calculate accuracy for this bin
        bin_accuracy = np.mean(predictions[in_bin] == true_labels[in_bin])
        
        # Calculate average confidence for this bin
        bin_confidence = np.mean(confidences[in_bin])
        
        # Calculate the weight of this bin (proportion of total predictions)
        bin_weight = np.mean(in_bin)
        
        # Add to ECE
        ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
    return ece
#%%
def delete_none_values(dataset):
    #delete any row with None value
    modified_dsets = {}
    for split_name in ['train', 'validation', 'test']:
        modified_dsets[split_name] = dataset[split_name].filter(lambda x: all(value is not None for value in x.values()))
    return modified_dsets
#%%

#merge all splits train, test, and validation into one split
def get_ece_score(outputs_path, model_name, dataset_name, mergeSplits= False):
        output_dataset = load_from_disk(f"{outputs_path}{dataset_name}/first/{model_name}")
        output_dataset = delete_none_values(output_dataset)
        if mergeSplits:
            merged_dataset = merge_dataset_splits(output_dataset)
        else:
            merged_dataset = output_dataset
        
        y_true = merged_dataset['ground_truth']
        y_pred = merged_dataset['prediction']
        confidences = merged_dataset['confidence']
        confidences = [float(c) / 100 for c in confidences]

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confidences = np.array(confidences)
        ece = calculate_ece(confidences, y_pred, y_true, n_bins=10)

        print(f"model: {model_name}    ece", ece)
        return ece

        
#%%
get_ece_score(data_folder, 'Llama-7B', 'emowoz', mergeSplits=True)
#%%

