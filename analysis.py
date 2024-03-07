
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import load_from_disk
from datasets import Dataset, DatasetDict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc
from itertools import cycle
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
def get_combined_data(dataset, method):
    ground_truth_all = []
    prediction_all = []
    confidence_all = []
    softmax_values_all = []
    confidence_array = None
    softmax_preds = None
    # Function to extract columns and append to lists
    for split in ['train', 'test', 'validation']:
        split_data = dataset[split]
        ground_truth_all.extend(split_data['ground_truth'])
        
        # if the dataset has confidence values, append them to the list
        prediction_all.extend(split_data['prediction'])
        if method == "zero":
            continue            
        elif method == "first":
            confidence_all.extend(split_data['confidence'])
        elif method == "logits":
            softmax_values_all.extend(split_data['softmax_transition'])
    # Convert lists to NumPy arrays
    ground_truth_array = np.array(ground_truth_all)
    prediction_array = np.array(prediction_all)
    if method == "logits":
        softmax_preds = np.array(softmax_values_all)
    if method == "zero":
        #do nothing
        pass
    elif method == "first":
        confidence_array = np.array(confidence_all)/100
    return ground_truth_array, prediction_array,  softmax_preds, confidence_array
#%%
#clead dataset
def datase_cleaning(dataset,  method,emotion2idx):
    ground_truth_set = emotion2idx.keys()
    #delete all instances where the prediction has value that is not in the ground truth and convert emotion to its index using emotion2_idx
    for split in ['train', 'test', 'validation']:
        if 'prediction_transition' in dataset[split].column_names:
            #change the name to prediction
            dataset[split] = dataset[split].rename_column('prediction_transition', 'prediction')
        # delete confidence column if it exists
        dataset[split] = dataset[split].filter(lambda x: x['prediction'] in ground_truth_set)
        dataset[split] = dataset[split].map(lambda x: {'ground_truth': emotion2idx[x['ground_truth']], 'prediction': emotion2idx[x['prediction']]})
        if method =="logits":
            # remove any instance where any item in the softmax_transition list is None
            dataset[split] = dataset[split].filter(lambda x: any(x['softmax_transition']))

    return dataset

#%%
#merge all splits train, test, and validation into one split
def get_ece_score( y_true, y_pred, confidence):
    ece = calculate_ece(y_true,y_pred, confidence, n_bins=10)
    print(f"dataset: {dataset_name} ,  model: {model_name}  , ece: {ece}")
    return ece
#%%
def get_ece_softmax(y_true, y_pred_probs, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for multiclass classification.
    
    Parameters:
    - y_true: numpy array of shape (n_samples,) with true class labels.
    - y_pred_probs: numpy array of shape (n_samples, n_classes) with predicted probabilities.
    - n_bins: Number of bins to use for calibration error calculation.
    
    Returns:
    - ece: The Expected Calibration Error.
    """
    # Convert true labels to one-hot encoding
    y_true_one_hot = np.eye(np.max(y_true) + 1)[y_true]

    # Get the predicted class and confidence for each sample
    pred_confidences = np.max(y_pred_probs, axis=1)
    pred_classes = np.argmax(y_pred_probs, axis=1)
    true_correct = (pred_classes == y_true).astype(float)

    # Initialize bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0  # Initialize ECE
    
    # Compute ECE by binning predictions
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of samples that fall into the current bin
        in_bin = (pred_confidences > bin_lower) & (pred_confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            # Average accuracy and confidence in the bin
            bin_accuracy = np.mean(true_correct[in_bin])
            bin_confidence = np.mean(pred_confidences[in_bin])
            
            # Weight of the bin (by the number of samples)
            bin_weight = np.mean(in_bin)
            
            # Accumulate weighted absolute difference
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
            
    return ece
#%%
def get_brier_score(y_true, prediction_probs):
    # One-hot encode the actual class labels
    num_classes = len(prediction_probs[0])
    y_true_one_hot = label_binarize(y_true, classes=range(num_classes))

    # Calculate the Brier score for each class
    brier_scores = [brier_score_loss(y_true_one_hot[:, i], prediction_probs[:, i]) for i in range(num_classes)]

    # Calculate the average Brier score across all classes
    average_brier_score = np.mean(brier_scores)

    print(f"Average Brier Score: {average_brier_score}")
    return average_brier_score

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

#%%


def plot_roc_curve(y_true, y_pred_probs,  ax):
    # Binarize the output labels in one-vs-rest fashion
    classes = np.arange(len(y_pred_probs[1]))
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # Compute ROC curve and ROC area for each class
    n_classes = y_true_bin.shape[1]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')


#%%
# main
datasets = ['meld', 'emowoz', 'emocx']
models = ['Llama-7B', 'Llama-13B', 'Mistral-7B', 'Zephyr-7B']

data_folder = f"/home/samad/Desktop/ACII/data/"

emotion_labels ={'meld': ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
, 'emowoz': ["neutral", "disappointed", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"], 
'emocx':["others", "happy", "sad" , "angry"]}
methods = ['zero', 'first','logits', 'ptrue']
method = methods[2]
fig, ax = plt.subplots()
for model_name in models:
    for dataset_name in datasets:
        emotion2idx = {v:k for k,v in enumerate (emotion_labels[dataset_name])}
        print(f"*************dataset: {dataset_name} ,  model: {model_name}, method: {method} ************* ")
        outputs_path = data_folder
        path = f"{outputs_path}{dataset_name}/{method}/{model_name}"
        # if pass is not available continue to the next dataset
        try:
            print (path)
            raw_dataset = load_from_disk(path)
        except:
            print(f"pass: {path} not available")
            continue        

        cleaned_dataset = datase_cleaning(raw_dataset,method, emotion2idx)
        y_true, y_pred, softmax_preds,confidence = get_combined_data(cleaned_dataset, method)
        f_scores = get_accuracy_scores(y_true, y_pred)
        if method == "zero":
                continue
        elif method == "first":
                ece = get_ece_score(y_true, y_pred, confidence)
                plot_calibration_diagram(y_true, y_pred, confidence)
        elif method == "logits":
            roc_auc = roc_auc_score(y_true, softmax_preds, multi_class="ovr", average="weighted")
            plot_roc_curve(y_true, softmax_preds,  ax)

            brier_score = get_brier_score(y_true, softmax_preds)
            ece = get_ece_softmax(y_true, softmax_preds)
            print(f"Weighted AUC-ROC score: {roc_auc}")
            print(f"brier_score: {brier_score}")
            print(f"ece: {ece}")
            

        print(f"f-scores: {f_scores}")
        plot_confusion_matrix(y_true, y_pred,emotion_labels[dataset_name])
ax.legend(loc='lower right')
plt.show()

# 



# %%

# %%
