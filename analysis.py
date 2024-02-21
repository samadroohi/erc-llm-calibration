#%%
from datasets import load_from_disk
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
#%%
ds_verbalized = load_from_disk("data/ed_verbalized_None_non-definitive_uncertainty_meld_all_splits")
# %%
#remove none-value instances
for split_name, dataset in ds_verbalized.items():
    # Operational sorting to deal with flanking out all instances that have `None`.
    # Dataset.map, here, is the clairvoyant's dance - powerful yet, might need a more specific form based on your data nature.
    ds_verbalized[split_name] = dataset.filter(lambda x: all(value is not None for value in x.values()))

# %%
f_score = f1_score(ds_verbalized['train']['ground_truth'], ds_verbalized['train']['prediction'], average='macro')
    
    
# %%
f_score
# %%
