
#%%
from datasets import load_from_disk
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
#%%


ds_pred_ndef_llama7b = load_from_disk("data/rawpredictions/non-definitve/Llama-7B-meld")
ds_pred_ndef_llama13b = load_from_disk("data/rawpredictions/non-definitve/Llama-13B-meld")
ds_pred_ndef_mistral= load_from_disk("data/rawpredictions/non-definitve/Mistral-7B-meld")
ds_pred_ndef_zephyr = load_from_disk("data/rawpredictions/non-definitve/Zephyr-7B-meld")

ds_pred_def_llama7b = load_from_disk("data/rawpredictions/definitve/Llama-7B-meld")
ds_pred_def_llama13b = load_from_disk("data/rawpredictions/definitve/Llama-13B-meld")
ds_pred_def_mistral= load_from_disk("data/rawpredictions/definitve/Mistral-7B-meld")
ds_pred_def_zephyr = load_from_disk("data/rawpredictions/definitve/Zephyr-7B-meld")


# %%
#remove none-value instances
def get_accuracy_scores(dset):
    modified_dsets = {}
    f_scores ={}
    for split_name in ['train', 'validation', 'test']:
        modified_dsets[split_name] = dset[split_name].filter(lambda x: all(value is not None for value in x.values()))
        f_scores[split_name] = f1_score(modified_dsets[split_name]['ground_truth'], modified_dsets[split_name]['prediction'], average='weighted')
    print(f"fscore", f_scores)
    return f_scores

# %%
f_score = get_accuracy_scores(ds_pred_ndef_zephyr)
f_score = get_accuracy_scores(ds_pred_ndef_mistral)
f_score = get_accuracy_scores(ds_pred_ndef_llama13b)
f_score = get_accuracy_scores(ds_pred_ndef_llama7b)
# %%
f_score = get_accuracy_scores(ds_pred_def_zephyr)
f_score = get_accuracy_scores(ds_pred_def_mistral)
f_score = get_accuracy_scores(ds_pred_def_llama13b)
f_score = get_accuracy_scores(ds_pred_def_llama7b)
# %%