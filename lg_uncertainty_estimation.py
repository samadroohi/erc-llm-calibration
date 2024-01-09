#%%
import torch,gc
import bitsandbytes as bnb
from transformers import LlamaTokenizer, AutoTokenizer
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, set_seed, \
    TrainingArguments, BitsAndBytesConfig, TrainingArguments
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
from datapreprocessing import load_ds, group_dialogues, extract_context_meld, extract_context_emowoz, extract_context_dailydialog
from huggingface_hub import login
import json
import re
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import requests
import templates

#%%

class LogitBasedUncertaitnyEstimation():
    def __init__(self, model, tokenizer, device):

        # Initialize parameters specific to Approach 1
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_prompts):
        all_outputs = []
        all_input_ids = []
        for prompt in input_prompts:
            input_ids = self.tokenizer(prompt, padding=True, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model(input_ids)
            logits = outputs.logits.detach().cpu()
            all_outputs.append(logits)
            all_input_ids.append(input_ids.detach().cpu())
            del outputs, input_ids
            torch.cuda.empty_cache()
        all_outputs = torch.concat(all_outputs, 0)[:, -1:, :]  # We take the logit corresponding to the option token
             
        all_input_ids = torch.concat(all_input_ids, 0)[:, -1:]  # We also include the token id for the options
        probs = torch.log_softmax(all_outputs.float(), dim=-1).detach().cpu()  # Log softmax scores
        torch.cuda.empty_cache()

        gen_probs = torch.gather(probs, 2, all_input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(all_input_ids[:, 0], gen_probs[:, 0]):
            batch.append((self.tokenizer.decode(input_sentence), input_probs.item()))
        return batch
    
def softmax(logits):
    '''
    converts log-softmax scores to probablities.
    '''
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp_logits
    return probabilities
def extract_answer(batch):
    '''
    converts the batch of option, log-softmax score tuples to option, probablity tuples
    '''
    probabilities = softmax(np.array([answer[-1] for answer in batch]))

    output_with_probabilities = [(batch[i][0], probabilities[i]) for i in range(len(batch))]
    return output_with_probabilities


        
        
def prepare_prompt(input_df, dataset_name):
    
    prompts = list()
    if dataset_name == 'emowoz':
            template = templates.template_emowoz
    elif dataset_name == 'meld':
        template = templates.template_meld
    elif dataset_name == 'dailydialog':
        template = templates.template_dailydialog
    for i in range (len(input_df['context'])):
        prompt = template(context=input_df['context'][i], query=input_df['query'][i]) 
        prompts.append(prompt)
    input_df['prompt_for_finetune'] = prompts

    return input_df
    
#%%
def extract_values(output):
    # Use regular expression to extract JSON-like part from the string
    data_str = output.split('Output:')[2]
    match = re.search(r'\{.*?\}', data_str, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Parse the JSON string
            data = json.loads(json_str)
            # Extract 'emotion' and 'confidence' values
            emotion = data.get('emotion', None)
            confidence = data.get('confidence', None)
            return emotion, confidence
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return None, None
    else:
        print("No JSON found in the string")
        return None, None
def model_settings(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True )
    
    model.model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token
    model.config.use_cache = False
    return model, tokenizer 
def save_split_to_dataset(split_name, split_df, dataset_dir):
    # Convert the DataFrame to a Hugging Face Dataset
    split_dataset = Dataset.from_pandas(split_df)

    # Load existing DatasetDict from disk if it exists, else create new
    if os.path.exists(dataset_dir):
        dataset_dict = load_from_disk(dataset_dir)
    else:
        dataset_dict = DatasetDict()

    # Update the DatasetDict with the new split
    dataset_dict[split_name] = split_dataset

    # Save the updated DatasetDict to disk
    dataset_dict.save_to_disk(dataset_dir)
    #Return a message
    return f"Saved {split_name} to {dataset_dir}"
#%%
def generate_responses(proccessed_data, split,model,tokenizer,device, mode, dataset_name):

    outputs = {'context':[], 'query':[], 'emotion':[], 'prompt_for_finetune':[], 'prediction':[],'confidence':[]}
    prompts_dataset = prepare_prompt(proccessed_data, dataset_name)
    for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
        if mode == "confidence-elicitation":
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            outputs_zero = model.generate(**inputs_zero, max_new_tokens=300, temperature=0.01)
            response = tokenizer.decode(outputs_zero[0], skip_special_tokens=False)
            output = extract_values(response)
            #print(f"output sequence: {output}")
        elif mode == "logit-based":
            #lg_ue = LogitBasedUncertaitnyEstimation(model, tokenizer, device)
            #output = lg_ue.forward([prompt_zero])
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            outputs=model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True,max_new_tokens=200)
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            print(f"output sequence: {tokenizer.decode(outputs.sequences[0])}")
            generated_tokens = outputs.sequences[:,input_length+1:input_length+2]
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
            # | token | token string | logits | probability
                print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy(force=True):.4f} | {np.exp(score.numpy(force=True)):.2%}")
            output = extract_answer(outputs)
            torch.cuda.empty_cache()
        outputs['context'].append(prompts_dataset['context'][i])
        outputs['query'].append(prompts_dataset['query'][i])
        outputs['emotion'].append(prompts_dataset['emotion'][i])
        outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
        outputs['prediction'].append(output[0]-1)
        outputs['confidence'].append(output[1])
        if i %1000 == 0 :
            print(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC")
            send_slack_notification(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC")
    return outputs
            
#%%    
def send_slack_notification(message):
    print(message + '\n')
    webhook_url = 'https://hooks.slack.com/services/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'  # Replace with your Webhook URL
    slack_data = {'text': message}

    response = requests.post(
        webhook_url, json=slack_data,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise ValueError(f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}")
#%%
def merge_datasets(data_path):
    merged_dataset = DatasetDict()
    for split in ['train', 'validation', 'test']:
        split_dataset = load_from_disk(f"{data_path}/{split}")
        merged_dataset[split] = split_dataset[split]
    merged_dataset.save_to_disk(f"{data_path}_all_splits")
    return f"Merged all splits into {data_path}"

def prepare_data(dataset_name, context_length):
    if dataset_name =='meld':
        datapath = {"train": "datasets/meld/train_sent_emo.csv", "validation": "datasets/meld/dev_sent_emo.csv", "test": "datasets/meld/test_sent_emo.csv"}
        datasets_df = load_ds(datapath)
        emotion_labels = datasets_df['train']['Emotion'].unique().tolist()
        emotion2idx = {v:k for k,v in enumerate (emotion_labels)}
        idx2emotion = {k:v for k,v in enumerate (emotion_labels)}
        ds_grouped_dialogues = group_dialogues(datasets_df)
        proccessed_data = {}
        proccessed_data['train'] = extract_context_meld(ds_grouped_dialogues['train'],context_length, emotion2idx)
        proccessed_data['test'] = extract_context_meld(ds_grouped_dialogues['test'],context_length, emotion2idx)
        proccessed_data['validation'] = extract_context_meld(ds_grouped_dialogues['validation'],context_length,emotion2idx)
    elif dataset_name =='emowoz':
        dataset = load_dataset("hhu-dsml/emowoz", 'emowoz')
        emotion_labels = ["unlabled","neutral", "fearful or sad/disappointed", "dissatisfied" , "apologetic", "abusive", "excited", "satisfied"]
        emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
        idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
        proccessed_data = {}
        proccessed_data['train'] = extract_context_emowoz(dataset['train'],context_length, idx2emotion)
        proccessed_data['test'] = extract_context_emowoz(dataset['test'],context_length, idx2emotion)
        proccessed_data['validation'] = extract_context_emowoz(dataset['validation'],context_length, idx2emotion)
    elif dataset_name =='dailydialog':
        dataset = load_dataset("daily_dialog")
        emotion_labels = ["neutral", "anger", "disgust" , "fear", "happiness", "sadness", "surprise"]
        emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
        idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
        proccessed_data = {}
        proccessed_data['train'] = extract_context_dailydialog(dataset['train'],context_length, idx2emotion)
        proccessed_data['test'] = extract_context_dailydialog(dataset['test'],context_length, idx2emotion)
        proccessed_data['validation'] = extract_context_dailydialog(dataset['validation'],context_length, idx2emotion)
    return proccessed_data, emotion2idx, idx2emotion

#%%
#Main
def main():

    gc.collect()
    torch.cuda.empty_cache()
    _ = load_dotenv(find_dotenv())
    datasets = ['meld', 'emowoz','dailydialog']
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    model, tokenizer = model_settings(model_name)#,device_map
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_name in datasets:
        send_slack_notification( f"The progam started for dataset: {dataset_name}")
        context_length = 2 # the maximum number of utterances to be considered for the context
        #%%
        #Load model
        #num_layers = 40 #for llama-2-7b-chat-hf, 40 for llama-2-13b-hf
        # device_map = {
        #     "model.embed_tokens": 0,
        #     "model.norm": 1,
        #     "lm_head": 1,
        # } | {
        #     f"model.layers.{i}": int(i >= 20) for i in range(num_layers)
        # }
        proccessed_data, emotion2idx, idx2emotion = prepare_data(dataset_name, context_length)
        #%%
        new_datapath = f'data/ed_verbalized_uncertainty_{dataset_name}'
        response = None
        splits = ['train', 'validation', 'test']
        modes = ["confidence-elicitation", "logit-based"]
        mode = modes[0]
        try:
            for split in splits: 
                outputs = generate_responses(proccessed_data[split],split,model,tokenizer, device, mode, dataset_name)
                new_df = pd.DataFrame(outputs)
                ds_path = f"{new_datapath}/{split}"
                save_split_to_dataset(split, new_df, ds_path)
                send_slack_notification( f"Uncertainty verbalization completed for split {dataset_name}:{split}")
            message= merge_datasets(new_datapath)
            send_slack_notification(f"Uncertainty verbalization completed successfully: {message}")
        except Exception as e:
            send_slack_notification(f"UERC failed with error: {e}")  

#%%
if __name__=="__main__":
    main()
    print("Finished successfully!")

#%%

# ds = load_from_disk("data/ed_verbalized_uncertainty_meld_all_splits")
# print(ds)
# # %%
# ds1 = load_from_disk("data/ed_verbalized_uncertainty_meld_all_splits")
# ds2 = load_from_disk("data/ed_verbalized_uncertainty_emowoz_all_splits")
# ds3 = load_from_disk("data/ed_verbalized_uncertainty_dailydialog_all_splits")
# # %%
# print(ds1['train'][1])
# print(ds2['train'][1])
# print(ds3['train'][1])

    #%%
#show the whole dataframe 


# %%
