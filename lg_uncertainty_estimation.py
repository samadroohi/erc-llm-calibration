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
import LlamaMeldTemplates as lmtemplate
import MistralMeldTemplates as mmtemplate
from utils import *
torch.cuda.empty_cache()

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


        
        
def prepare_prompt(input_df, dataset_name ,mode, model_template,template_type=None,inserted_emotion=None ):
    
    prompts = list()
    if dataset_name == 'emowoz':
            template = model_template.template_emowoz
    elif dataset_name == 'meld':
        if template_type == "non-definitive":  
            template = model_template.template_meld_ndef
        elif template_type == "definitive":
            template = model_template.template_meld_def
    elif dataset_name == 'dailydialog':
        template = model_template.template_dailydialog
    for i in range (len(input_df['context'])):
        #print(f"input_df['context'][i]: {input_df['context'][i]}, input_df['query'][i]: {input_df['query'][i]}, input_df['emotion'][i]: {input_df['emotion'][i]}")
        context=input_df['context'][i] 
        query=input_df['query'][i]
        if mode == "P(True)":
            prompt = template(context, query, mode, emotion_label=inserted_emotion[i])
        else:
            prompt = template(context, query, mode)
        prompts.append(prompt)
    input_df['prompt_for_finetune'] = prompts

    return input_df
    
#%%
#This method is used for extracting emotion and confidence from the model output
def extract_values(output_str):
    pattern = r"\{.*?\}"

# Find all JSON strings in the text
    json_strings = re.findall(pattern, output_str, re.DOTALL)

    # Check the number of found JSON strings
    if len(json_strings) >= 3:
        # Parse the third JSON string
        third_json_obj = json.loads(json_strings[2])

        # Extract 'emotion' and 'confidence' values
        emotion = third_json_obj.get('emotion')
        index = third_json_obj.get('index')
        confidence = third_json_obj.get('confidence')

        #print(f"Emotion: {emotion}, Index:{index} Confidence: {confidence}")
    else:
        emotion = None
        confidence = None
        index = None
        #log error
        print(f"Expected at least 3 JSON objects, but found {len(json_strings)}.")
    return (emotion,index, confidence)

def model_settings(model_name, num_layers ,single_gpu): #, device_map
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if single_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
    else:
        device_map = {
                "model.embed_tokens": 0,
                "model.norm": 1,
                "lm_head": 1,
            } | {
                f"model.layers.{i}": int(i >= 20) for i in range(num_layers)
            }
        model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config, device_map=device_map) #

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
def get_transition_scores(inputs_zero,model, tokenizer, outputs_gen, max_new_tokens):
    token_logits = { "token_string":[], "probability":[]}
    #print("########Scores for transition########")
    #print(f"shape of scores is: {scores.shape}")
    transition_scores = model.compute_transition_scores(outputs_gen.sequences, outputs_gen.scores, normalize_logits=True)    
    #print(f"shape of transition_scores is: {transition_scores}")
    input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
    generated_tokens = outputs_gen.sequences[:, input_length:]
    #generated text
    #print(f"Generated text: {tokenizer.decode(generated_tokens[0])}")
    #print(f"length of generated tokens is: {len(generated_tokens[0])}  and length of transition scores is : {len(transition_scores[0])}")
  
    for tok, score in zip(generated_tokens[0][:max_new_tokens], transition_scores[0][:max_new_tokens]):
        # | token | token string | logits | probability | {np.exp(score.numpy()):.2}%
        #token_logits["token"].append(tok)
        token_logits["token_string"].append(tokenizer.decode(tok))
        #token_logits["logits"].append(score.numpy())
        token_logits["probability"].append(np.exp(score.numpy()))
        #print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2}%")
    return token_logits
def get_logits_for_generated_output(generated_output, model, tokenizer,input_length, max_new_tokens):
    #print("########Scores for logits1 ########")
    # Take the text generated and re-evaluate the probability 
    token_logits = {"token":[], "token_string":[], "logits":[], "probability":[]}
    text_generated = tokenizer.batch_decode(generated_output.sequences, skip_special_tokens= True)[0]
    generated_output_ids = tokenizer(text_generated, return_tensors="pt").input_ids

    with torch.no_grad():
        model_output = model(generated_output_ids)
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = torch.log_softmax(model_output.logits, dim=-1).detach()
    probs = probs[:, :-1, :]
    generated_input_ids_shifted = generated_output_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, generated_input_ids_shifted[:, :, None]).squeeze(-1)
    #print(gen_probs[:,:max_new_tokens])
    for tok, score in zip(generated_input_ids_shifted[0][input_length:input_length+max_new_tokens], gen_probs[0][input_length:input_length+max_new_tokens]):
        #token_logits["token"].append(tok)
        token_logits["token_string"].append(tokenizer.decode(tok))
        #token_logits["logits"].append(score.numpy())
        token_logits["probability"].append(np.exp(score.numpy()))
        #print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
    return token_logits
def extract_label_probs(token_logits, target_tokens):
    label = None
    label_prob=-1
    for i, token in enumerate(zip(token_logits["token_string"], token_logits["probability"])):
        if token[0] in target_tokens:
            label = token[0]
            label_prob = token[1]

    return (label,label_prob)
def extract_label(output_strings):
    matches = re.findall(r'\[(.*?)\]', output_strings)
    if len(matches) == 1:
        return matches[0]
    else:
        print(f"Expected 1 match, but found {len(matches)}.")
    
    

#%%
def generate_responses(proccessed_data, split,model,tokenizer,device, mode,  dataset_name, model_template, error_flag, emotion_tokens, idx2emotion, assess_type=None, template_type=None):

    outputs = {'context':[], 'query':[], 'ground_truth':[], 'prompt_for_finetune':[]}
  
    if mode == 'verbalized':
        prompts_dataset = prepare_prompt(proccessed_data, dataset_name, mode, model_template,template_type= template_type)
        outputs['prediction']= []
        #outputs['confidence'] = []
        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs_zero.sequences[0][input_length:], skip_special_tokens=False)
            #print(f"output sequence: {response}. ground truth: {prompts_dataset['emotion'][i]}")
            output = extract_label(response)
            #print(f"output: {output}")
            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            outputs['ground_truth'].append(prompts_dataset['emotion'][i])
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            outputs['prediction'].append(output)
            #outputs['confidence'].append(output[2])
            if i %100 == 1:
            #print(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC ")
            #send_slack_notification(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC", error_flag)
                print( "Query: " , outputs['query'][i], ",      ground truth: ", outputs['ground_truth'][i], ",     prediction: ", 
                    outputs['prediction'][i])

      
                
    elif mode == "logit-based":
        prompts_dataset = prepare_prompt(proccessed_data, dataset_name, mode)
        outputs['prediction_label']=[]
        outputs['prediction_emotion_model']=[]
        outputs['confidence_model']=[]
        outputs['prediction_emotion_transition']=[]
        outputs['confidence_transition']=[]
        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=100)
            response = tokenizer.decode(outputs_zero.sequences[0], skip_special_tokens=False)
            #print(f"output sequence: {response}")
            transition_scores = get_transition_scores(inputs_zero,model, tokenizer ,outputs_zero, 6)
            model_scores = get_logits_for_generated_output(outputs_zero, model, tokenizer,input_length, 6)
            label_probs_transition = extract_label_probs(transition_scores, emotion_tokens)
            label_probs_model = extract_label_probs(model_scores, emotion_tokens)
            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            outputs['ground_truth'].append(idx2emotion[prompts_dataset['ground_truth'][i]])
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            outputs['prediction_emotion_model'].append(label_probs_model[0])
            outputs['confidence_model'].append(label_probs_model[1])
            outputs['prediction_emotion_transition'].append(label_probs_transition[0])
            outputs['confidence_transition'].append(label_probs_transition[1])
            #if i % 1000 == 1:
            print(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC")
            send_slack_notification(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC", error_flag)
            print( "Query: " , outputs['query'][i], ",   ground truth: ", outputs['ground_truth'][i], ",  prediction_emotion_model: ", 
                      outputs['prediction_emotion_model'][i], "   , confidence_model:",  outputs['confidence_model'][i],', prediction_emotion_transition:',outputs['prediction_emotion_transition'][i]
                        ,', confidence_transition: ', outputs['confidence_transition'][i] )
            
    elif mode == "P(True)":
        if assess_type == "self-assessment":
            inserted_emotion = proccessed_data['prediction_emotion']
        elif assess_type == "random-assessment":
            inserted_emotion = np.random.choice(list(idx2emotion.values()), len(proccessed_data['context']))
        prompts_dataset = prepare_prompt(proccessed_data, dataset_name,mode,
                                         template_type= template_type, 
                                         inserted_emotion = inserted_emotion 
                                         )
        outputs['emotion_inserted']=[]
        outputs['prediction_truthfulness']=[] #A:True B:False
        outputs['ptrue-transition_probs']=[]
        outputs['ptrue-model_probs']=[]
        target_tokens = ['A', 'B']
        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt")
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=200)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            response = tokenizer.decode(outputs_zero.sequences[0][input_length:], skip_special_tokens=False)
            
            transition_scores = get_transition_scores(inputs_zero, model, tokenizer ,outputs_zero, 4)
            model_scores = get_logits_for_generated_output(outputs_zero, model, tokenizer,input_length, 4)
            label_probs_transition = extract_label_probs(transition_scores, target_tokens)            
            label_probs_model = extract_label_probs(model_scores, target_tokens)

            

            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            outputs['ground_truth'].append(prompts_dataset['ground_truth'][i])
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            outputs['emotion_inserted'].append(inserted_emotion[i])
            outputs['prediction_truthfulness'].append(label_probs_model[0])
            outputs['ptrue-transition_probs'].append(label_probs_transition[1])
            outputs['ptrue-model_probs'].append(label_probs_model[1])
            if i %100 == 1:
                print(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC ")
                send_slack_notification(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC", error_flag)
                if label_probs_model[0] == "B":
                    print(f"\n ******\n output sequence: {response} ")
                print( "Query: " , outputs['query'][i], ",   ground truth: ", outputs['ground_truth'][i],  "emotion_inserted:", outputs["emotion_inserted"][i], ", prediction_truthfulness: ", 
                    outputs['prediction_truthfulness'][i], "   , ptrue-transition_probs:",  outputs['ptrue-transition_probs'][i],', ptrue-model_probs:',outputs['ptrue-model_probs'][i] )
                
    return outputs
    
            
#%%    
def send_slack_notification(message, error_flag):
    if error_flag:
        print(message + '\n')
        webhook_url = 'https://hooks.slack.com/services/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'  # Replace with your Webhook URL
        slack_data = {'text': message}

        response = requests.post(
            webhook_url, json=slack_data,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code != 200:
            raise ValueError(f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}")
    else:
        print(message + '\n')
#%%
def merge_datasets(data_path):
    merged_dataset = DatasetDict()
    for split in ['train', 'validation', 'test']:
        split_dataset = load_from_disk(f"{data_path}/{split}")
        merged_dataset[split] = split_dataset[split]
    merged_dataset.save_to_disk(f"{data_path}_all_splits")
    return f"Merged all splits into {data_path}"

def prepare_data(dataset_name, context_length, mode, assess_type):
    if dataset_name =='meld':
        emotion2idx= {'neutral': 0,'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        emotion_labels = emotion2idx.keys()
        idx2emotion = {k:v for k,v in enumerate (emotion_labels)}
        proccessed_data = {}

        if mode == "verbalized":
            datapath = {"train": "datasets/meld/train_sent_emo.csv", "validation": "datasets/meld/dev_sent_emo.csv", "test": "datasets/meld/test_sent_emo.csv"}
            datasets_df = load_ds(datapath)
            ds_grouped_dialogues = group_dialogues(datasets_df)
            proccessed_data['train'] = extract_context_meld(ds_grouped_dialogues['train'],context_length)
            proccessed_data['test'] = extract_context_meld(ds_grouped_dialogues['test'],context_length)
            proccessed_data['validation'] = extract_context_meld(ds_grouped_dialogues['validation'],context_length)

        elif mode == "P(True)":
            dataset_dict = load_from_disk(f"data/ed_verbalized_uncertainty_{dataset_name}_all_splits")
            if assess_type == "self-assessment":
                features = ['context', 'query','ground_truth', 'prediction_emotion']
            elif assess_type == "random-assessment":
                features = ['context', 'query','ground_truth']
            for split in ['train', 'validation', 'test']:
                proccessed_data[split] = dataset_dict[split].to_pandas()
                proccessed_data[split] = proccessed_data[split].drop(columns=[col for col in proccessed_data[split].columns if col not in features])


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

#%%
#Main
#def main():
singleGPU = True
error_flag = False
gc.collect()
torch.cuda.empty_cache()
_ = load_dotenv(find_dotenv())
datasets = ['meld'] #Add 'emowoz' and 'dailydialog' to the list
models = ["meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]
model_template = mmtemplate #mmtemplate for misteralmeld , and lmtemplate for lamameld
model_name = models[1]
num_layers = 32 #32 for 7B and 40 for 13B

#Load model


model, tokenizer = model_settings(model_name,num_layers, singleGPU) #,device_map

dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")
device = dev1 if torch.cuda.device_count() > 1 else dev0
emotion_tokens = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger", "dis", "sad", "Ang", "Ne", "Jo", "S", "Dis", "Sur", "F"]
#emotion_tokens_13b = 


modes = ["verbalized", "logit-based", "P(True)"]
mode = modes[0]

assess_type=None
if mode == "P(True)":
    assess_types = ["self-assessment", "random-assessment"] #  results from the verbalized prediction, random labels,
    assess_type = assess_types[0] #self-assessment is for computing P(True) on the results generated from the verbalization method

template_types = ["non-definitive", "definitive"]
template_type = template_types[0]
#%%
for dataset_name in datasets:
    send_slack_notification( f"The progam started for dataset: {dataset_name}", error_flag)
    context_length = 2 # the maximum number of utterances to be considered for the context
    proccessed_data, emotion2idx, idx2emotion = prepare_data(dataset_name, context_length, mode,assess_type)
    new_datapath = f'data/ed_{mode}_{assess_type}_{template_type}_uncertainty_{dataset_name}_{model_name}'
    #print(proccessed_data['train'].head(1))
    response = None
    splits = ['train', 'validation', 'test']
    try:
        for split in splits:
            print(f"************Started {split} for dataset {dataset_name}**********") 
            outputs = generate_responses(proccessed_data[split],
                                         split,model,tokenizer, device,
                                           mode, dataset_name, model_template,
                                           error_flag,emotion_tokens,idx2emotion, 
                                             assess_type=assess_type,
                                             template_type= template_type)
            new_df = pd.DataFrame(outputs)
            ds_path = f"{new_datapath}/{split}"
            save_split_to_dataset(split, new_df, ds_path)
            send_slack_notification( f"Uncertainty {mode} completed for split {dataset_name}:{split}", error_flag)
        message= merge_datasets(new_datapath)
        send_slack_notification(f"Uncertainty {mode} completed successfully: {message}", error_flag)
    except Exception as e:
        send_slack_notification(f"UERC failed with error: {e}", error_flag)  

#%%
# if __name__=="__main__":
#     main()
#     print("Finished successfully!")

#%%

#ds1 = load_from_disk("data/ed_verbalized_uncertainty_meld_all_splits")
#print(ds1['train'])

# # %%
#features = ['context', 'query', 'ground_truth', 'prompt_for_finetune', 'prediction_emotion', 'confidence']
#ind = 1
#print('context: ' , ds1['train']['context'][ind], 'query: ', ds1['train']['query'][ind], 'ground_truth: ', ds1['train']['ground_truth'][ind], 'prompt_for_finetune: ', ds1['train']['prompt_for_finetune'][ind], 'prediction_emotion: ', ds1['train']['prediction_emotion'][ind], 'confidence: ', ds1['train']['confidence'][ind]  )


    #%%

ds1 = load_from_disk("data/ed_verbalized_None_non-definitive_uncertainty_meld_mistralai/Mistral-7B-Instruct-v0.2_all_splits")
ds1['test'][0]
# %%