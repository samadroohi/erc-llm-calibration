
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import load_from_disk
from datasets import Dataset, DatasetDict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%%


#%%


# %%
#remove none-value instances
def get_accuracy_scores( y_true,y_pred):
    f_scores ={}
    f_score = f1_score(y_true, y_pred, average='weighted')
    f_score_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    f_scores['f1'] = f_score
    f_scores['f1_macro'] = f_score_macro
    f_scores['precision'] = precision
    f_scores['recall'] = recall
    f_scores['accuracy'] = accuracy
    return f_scores
#%%
def plot_confusion_matrix(y_true, y_pred, labels):
    
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#%%

from sklearn.preprocessing import OneHotEncoder

def calculate_ece(y_true, y_pred, confidences, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for multiclass classification
    with confidence values only for the predicted class.
    
    Parameters:
    - y_true: Array of true labels.
    - y_pred: Array of predicted labels.
    - confidences: Array of confidence values for the predicted labels.
    - n_bins: Number of bins to divide the confidence scores.
    
    Returns:
    - ece: The Expected Calibration Error.
    """
    bin_limits = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]
    
    ece = 0.0  # Expected Calibration Error
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of samples whose confidence fall into the current bin
        in_bin = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        if len(in_bin) == 0:
            continue
        
        # Calculate accuracy for this bin
        bin_accuracy = np.mean(y_true[in_bin] == y_pred[in_bin])
        
        # Calculate average confidence for this bin
        bin_confidence = np.mean(confidences[in_bin])
        
        # Proportion of samples in this bin
        bin_weight = len(in_bin) / len(y_true)
        
        # Contribution of this bin to the ECE
        ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
        
    return ece
#%%
def get_combined_data(dataset, stage_of_verbalization):
    ground_truth_all = []
    prediction_all = []
    confidence_all = []
    confidence_array = None
    # Function to extract columns and append to lists
    for split in ['train', 'test', 'validation']:
        split_data = dataset[split]
        ground_truth_all.extend(split_data['ground_truth'])
        prediction_all.extend(split_data['prediction'])
        # if the dataset has confidence values, append them to the list
        if stage_of_verbalization == "first":
            confidence_all.extend(split_data['confidence'])
            


    # Convert lists to NumPy arrays
    ground_truth_array = np.array(ground_truth_all)
    prediction_array = np.array(prediction_all)
    if stage_of_verbalization == "first":
        confidence_array = np.array(confidence_all)/100
    return ground_truth_array, prediction_array, confidence_array
#%%
#clead dataset
def datase_cleaning(dataset, emotion_labels):
    emotion2idx = {v:k for k,v in enumerate (emotion_labels)}
    ground_truth_set = emotion2idx.keys()
    #delete all instances where the prediction has value that is not in the ground truth and convert emotion to its index using emotion2_idx
    for split in ['train', 'test', 'validation']:
        # delete confidence column if it exists
        dataset[split] = dataset[split].filter(lambda x: x['prediction'] in ground_truth_set)
        dataset[split] = dataset[split].map(lambda x: {'ground_truth': emotion2idx[x['ground_truth']], 'prediction': emotion2idx[x['prediction']]})
    return dataset

#%%
#merge all splits train, test, and validation into one split
def get_ece_score( y_true, y_pred, confidence):
    ece = calculate_ece(y_true,y_pred, confidence, n_bins=10)
    print(f"dataset: {dataset_name} ,  model: {model_name}  , ece: {ece}")
    return ece

        
#%%
def plot_calibration_diagram(y_true, y_pred, confidence):
    # Determine correctness of each prediction
    correct = (y_true == y_pred).astype(int)

    # Bin the data & calculate the fraction of correct predictions per bin
    bins = np.linspace(0, 1, 11)
    digitized = np.digitize(confidence, bins) - 1  # Bin indices for each prediction
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + bin_width / 2  # Midpoints of bins

    fraction_of_positives = np.array([correct[digitized == i].mean() for i in range(len(bins)-1)])
    bin_centers = bin_centers[~np.isnan(fraction_of_positives)]
    fraction_of_positives = fraction_of_positives[~np.isnan(fraction_of_positives)]

    # Plot
    plt.plot(bin_centers, fraction_of_positives, 's-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Confidence Score')
    plt.ylabel('Fraction of Correct Predictions')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()

# %%
#%%
# main
datasets = ['meld', 'emowoz', 'emocx']
models = ['Llama-7B', 'Llama-13B', 'Mistral-7B', 'Zephyr-7B']

data_folder = f"/home/samad/Desktop/ACII/data/"

emotion_labels ={'meld': ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
, 'emowoz': ["neutral", "disappointed", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"], 
'emocx':["others", "happy", "sad" , "angry"]}
stage_of_verbalization = 'first'
for model_name in models:
    for dataset_name in datasets:
        print(f"dataset: {dataset_name} ,  model: {model_name}")
        outputs_path = data_folder
        raw_dataset = load_from_disk(f"{outputs_path}{dataset_name}/{stage_of_verbalization}/{model_name}")
        cleaned_dataset = datase_cleaning(raw_dataset, emotion_labels[dataset_name])
        y_true, y_pred, confidence = get_combined_data(cleaned_dataset, stage_of_verbalization)
        f_scores = get_accuracy_scores(y_true, y_pred)
        if stage_of_verbalization == "first":
    
    #ece = get_ece_score(y_true, y_pred, confidence)
            plot_calibration_diagram(y_true, y_pred, confidence)

        print(f"f-scores: {f_scores}")
#print(ece)
        plot_confusion_matrix(y_true, y_pred,emotion_labels[dataset_name])


# 



# %%
