
#%%
from datasets import load_from_disk
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
#%%


ds_pred_ndef_llama7b = load_from_disk("data/rawpredictions/non-definitive/Llama-7B-meld")
ds_pred_ndef_llama13b = load_from_disk("data/rawpredictions/non-definitive/Llama-13B-meld")
ds_pred_ndef_mistral= load_from_disk("data/rawpredictions/non-definitive/Mistral-7B-meld")
ds_pred_ndef_zephyr = load_from_disk("data/rawpredictions/non-definitive/Zephyr-7B-meld")

ds_pred_def_llama7b = load_from_disk("data/rawpredictions/definitive/Llama-7B-meld")
ds_pred_def_llama13b = load_from_disk("data/rawpredictions/definitive/Llama-13B-meld/")
ds_pred_def_mistral= load_from_disk("data/rawpredictions/definitive/Mistral-7B-meld")
ds_pred_def_zephyr = load_from_disk("data/rawpredictions/definitive/Zephyr-7B-meld")

#%%
ds_pred_def_llama13b['train'][:10]


# %%
#remove none-value instances
def get_accuracy_scores(dset, model_name):
    modified_dsets = {}
    f_scores ={}
    for split_name in ['train', 'validation', 'test']:
        modified_dsets[split_name] = dset[split_name].filter(lambda x: all(value is not None for value in x.values()))
        f_scores[split_name] = f1_score(modified_dsets[split_name]['ground_truth'], modified_dsets[split_name]['prediction'], average='weighted')
    print(f"model: {model_name}    fscore", f_scores)
    return f_scores

# %%
f_score = get_accuracy_scores(ds_pred_ndef_zephyr, "zephyr-ndef")
f_score = get_accuracy_scores(ds_pred_ndef_mistral, "mistral-ndef")
f_score = get_accuracy_scores(ds_pred_ndef_llama13b, "llama13b-ndef")
f_score = get_accuracy_scores(ds_pred_ndef_llama7b, "llama7b-ndef")
# %%
f_score = get_accuracy_scores(ds_pred_def_zephyr, "zephyr-def")
f_score = get_accuracy_scores(ds_pred_def_mistral, "mistral-def")
f_score = get_accuracy_scores(ds_pred_def_llama13b, "llama13b-def")
f_score = get_accuracy_scores(ds_pred_def_llama7b, "llama7b-def")
# %%