#%%
from datasets import load_from_disk
#%%
ds_verbalized = load_from_disk("data/ed_verbalized_None_non-definitive_uncertainty_meld_all_splits")
ds_ptrue_self_assess_ndef = load_from_disk("data/ed_P(True)_self-assessment_non-definitive_uncertainty_meld_all_splits")
ds_ptrue_self_assess_def =  load_from_disk("data/ed_P(True)_self-assessment_definitive_uncertainty_meld_all_splits")
ds_logit = load_from_disk("data/ed_logit_uncertainty_meld_all_splits")
# %%

