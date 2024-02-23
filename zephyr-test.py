#%%
import torch 
from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig
#%%
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
bnb_config.max_length = 1024
bnb_config.temperature = 0.0001
bnb_config.do_sample = True
model_name = 'HuggingFaceH4/zephyr-7b-beta'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map = 'auto'
                                              ,quantization_config=bnb_config)
#%%
llm = pipeline('text-generation', model=model, 
               tokenizer=tokenizer,
               return_full_text=True,
               num_return_sequences=1,
               eos_token_id=tokenizer.eos_token_id,
               pad_token_id = tokenizer.pad_token_id)



# %%
system_prompt = """You are an emotion recognition assistant. You always analyse the context and query and recognitize the emotional state of the query utterence"""
messages = [{"role": "system",  "content": system_prompt}, {"role": "user", "content": "I am feeling happy"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# %%
output = llm(prompt)
# %%
output[0]['generated_text']
# %%
prompt = '<|system|>\nYou are an emotion recognition assistant. You always analyse the context and query and recognitize the emotional state of the query utterence</s>\n<|user|>\nI am feeling happy</s>\n<|assistant|>\nBased'
token_prompt = tokenizer(prompt, return_tensors='pt')

output =  model.generate(**token_prompt, max_length=1024, temperature=0.0001, do_sample=True)
# %%

print(tokenizer.decode(output[0], skip_special_tokens=True))
# %%
