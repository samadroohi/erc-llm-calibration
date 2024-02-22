
#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import MistralMeldTemplates
#%%
dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")

device = dev1 if torch.cuda.device_count() > 1 else dev0

num_layers = 32
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device_map = {
    "model.embed_tokens": 0,
    "model.norm": 1,
    "lm_head": 1,
}|{f"model.layers.{i}": int(i>=20) for i in range(num_layers)}

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote = True)

context = """[Monica]: This is disgusting! [disgust] 
            [Joey]: I also eat your food! [neutral]"""
query = "[Monica]: How dare you!"

prompt = MistralMeldTemplates.template_meld_ndef(context, query, "verbalized")

#messages = "What is the emotion label of this sentence: This is a hot day."

input_zero = tokenizer(prompt, return_tensors="pt").to(device)

generated_ids = model.generate(**input_zero, max_new_tokens=100, do_sample=True)
decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# %%
print(decoded)
# %%
print(prompt)
# %%
