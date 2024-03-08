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
from datapreprocessing import load_ds, group_dialogues, extract_context_meld, extract_context_emowoz, extract_context_dailydialog, extract_context_Emocx
from huggingface_hub import login
import json
import re
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import requests
from prompts.meld import LlamaMeldTemplates as lmtemplate
from prompts.meld import MistralMeldTemplates as mmtemplate
from prompts.meld import ZephyerMeldTemplates as zmtemplate
from prompts.emowoz import LlamaEmoWOZTemplates as letemplate
from prompts.emowoz import MistralEmoWOZTemplates as metemplate
from prompts.emowoz import ZephyerEmoWOZTemplates as zetemplate
from prompts.EmoContext import LlamaEmoCxTemplates as lcxtemplate
from prompts.EmoContext import MistralEmoCxTemplates as mcxtemplate
from prompts.EmoContext import ZephyerEmoCxTemplates as zcxtemplate
import os
 

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


        
        
def prepare_prompt(input_df, dataset_name ,mode, model_template,tokenizer, inserted_emotion=None, stage_of_verbalization=None ):
    
    prompts = list()
    if dataset_name == 'emowoz':
            template = model_template.template_emowoz
    elif dataset_name == 'meld':
        template = model_template.template_meld
    elif dataset_name == 'dailydialog':
        template = model_template.template_dailydialog
    elif dataset_name == 'emocx':
        template = model_template.template_emocx
    for i in range (len(input_df['context'])):
        #print(f"input_df['context'][i]: {input_df['context'][i]}, input_df['query'][i]: {input_df['query'][i]}, input_df['emotion'][i]: {input_df['emotion'][i]}")
        context=input_df['context'][i] 
        query=input_df['query'][i]
        if mode == "P(True)":
            prompt = template(context, query, mode,tokenizer, emotion_label=inserted_emotion[i])
        elif mode == "verbalized":
            prompt = template(context, query, mode,tokenizer=tokenizer, stage_of_verbalization=stage_of_verbalization)
        elif mode == "logit-based":
            prompt = template(context, query, mode,tokenizer=tokenizer)

        prompts.append(prompt)
    input_df['prompt_for_finetune'] = prompts

    return input_df
    
#%%


def model_settings(model_name): #, device_map
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=bnb_config, 
                                                     device_map='auto') #

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
def get_transition_scores( outputs_gen, tokens_dict):
    class_scores=[] 
    for token in tokens_dict.values():
        class_scores += [outputs_gen.scores[0][0][token]]
        softmax_scores = torch.softmax(torch.tensor(class_scores), dim=0).detach().cpu().numpy()
    predicted_label = list(tokens_dict.keys())[np.argmax(softmax_scores)]
    return softmax_scores, predicted_label

def get_model_scores(generated_output, model, tokenizer,tokens_dict, num_new_tokens):
    text_generated = tokenizer.batch_decode(generated_output.sequences, skip_special_tokens= True)[0]
    generated_output_ids = tokenizer(text_generated, return_tensors="pt").input_ids
    with torch.no_grad():
        model_output = model(generated_output_ids)
    emotion_tokens = list(tokens_dict.values())
    #-2 because the output starts with <s> special token therefore we always have one more token
    emotion_class_logits = [model_output.logits[0, -2, token] for token in emotion_tokens]
    softmax_scores = torch.softmax(torch.tensor(emotion_class_logits), dim=0).detach().cpu().numpy()
    softmax_scores = np.round(softmax_scores, 2)
    predicted_label = list(tokens_dict.keys())[np.argmax(softmax_scores)]
    return softmax_scores, predicted_label
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
        print(f"Output string: {output_strings}")
    
#This method is used for extracting emotion and confidence from the model output
def extract_lable_confidence(output_str):
    pattern = r"\{.*?\}"

# Find all JSON strings in the text
    json_strings = re.findall(pattern, output_str, re.DOTALL)

    # Check the number of found JSON strings
    if len(json_strings) == 1:
        # Parse the third JSON string
        third_json_obj = json.loads(json_strings[0])

        # Extract 'emotion' and 'confidence' values
        emotion = third_json_obj.get('prediction')
        confidence = third_json_obj.get('confidence')

        #print(f"Emotion: {emotion}, Index:{index} Confidence: {confidence}")
    else:
        emotion = None
        confidence = None
        #log error
        print(f"Expected exactly 1 JSON objects, but found {len(json_strings)}.")
        print(f"Output string: {output_str}")

    return (emotion, confidence)  

#%%
def generate_responses(processed_data, split,model,tokenizer,device, mode,  dataset_name, model_template, error_flag, tokens_dict, idx2emotion, assess_type=None, stage_of_verbalization=None):
    if stage_of_verbalization == "first":
        num_new_tokens = 25
    else:
        num_new_tokens = 1
    outputs = {'context':[], 'query':[], 'ground_truth':[], 'prompt_for_finetune':[]}
  
    if mode == 'verbalized':
        prompts_dataset = prepare_prompt(processed_data, dataset_name, mode, 
                                         model_template,
                                         tokenizer=tokenizer,
                                         stage_of_verbalization = stage_of_verbalization)
        outputs['prediction']= []
        outputs['confidence'] = []
        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=num_new_tokens, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs_zero.sequences[0][input_length:], skip_special_tokens=False)
            #print(f"output sequence: {response}. ground truth: {prompts_dataset['emotion'][i]}")
            if stage_of_verbalization == "zero":
                output = extract_label(response)
            elif stage_of_verbalization == "first":
                output = extract_lable_confidence(response) # This returns a tuple of (emotion, confidence)
            else:
                output = None # define a function to extract emotion and confidence from the output

            #print(f"output: {output}")
            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            if dataset_name == 'emowoz':
                outputs['ground_truth'].append(idx2emotion[prompts_dataset['emotion'][i]+1])
            else:            
                outputs['ground_truth'].append(prompts_dataset['emotion'][i])
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            
            if stage_of_verbalization != "zero":
                outputs['prediction'].append(output[0])
                outputs['confidence'].append(output[1])
            else:
                outputs['prediction'].append(output)
                outputs['confidence'].append(None)

            if i %100 == 1:
            #print(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC ")
            #send_slack_notification(f"Finished {i} out of {len(proccessed_data['context'])} for the split {split} for UERC", error_flag)
                print( "Query: " , outputs['query'][i], ",      ground truth: ", outputs['ground_truth'][i], ",     prediction: ", 
                      outputs['prediction'][i], "   , confidence:",  outputs['confidence'][i])
            torch.cuda.empty_cache()

       
    elif mode == "logit-based":
        prompts_dataset =  prepare_prompt(processed_data, dataset_name, mode, 
                                         model_template,
                                         tokenizer=tokenizer)
        #outputs['prediction_model']=[]
        #outputs['softmax_model']=[]
        outputs['softmax_transition']=[]
        outputs['prediction_transition']=[]
        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt").to(device)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=num_new_tokens, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs_zero.sequences[0][input_length:], skip_special_tokens=False)

            softmax_transition, prediction_transition = get_transition_scores(outputs_zero, tokens_dict)
            #softmax_model, prediction_model = get_model_scores(outputs_zero, model, tokenizer, emotion_tokens_dict, num_new_tokens)

            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            if dataset_name == 'emowoz':
                outputs['ground_truth'].append(idx2emotion[prompts_dataset['emotion'][i]+1])
            else:            
                outputs['ground_truth'].append(prompts_dataset['emotion'][i])            
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            #outputs['prediction_model'].append(prediction_model)
            #outputs['softmax_model'].append(softmax_model)
            outputs['prediction_transition'].append(prediction_transition)
            outputs['softmax_transition'].append(softmax_transition)
            if i % 100 == 1:
            #print(f"Finished {i} out of {len(processed_data['context'])} for the split {split} for UERC ")
                print( "Query: " , outputs['query'][i], ",   ground truth: ", outputs['ground_truth'][i], ", prediction_transition:", prediction_transition, ", softmax_transition:", softmax_transition)
            torch.cuda.empty_cache()
            
    elif mode == "P(True)":
        if assess_type == "self-assessment":
            inserted_emotion = processed_data['prediction']
        elif assess_type == "random-assessment":
            inserted_emotion = np.random.choice(list(idx2emotion.values()), len(processed_data['context']))
        prompts_dataset = prepare_prompt(processed_data, dataset_name,mode,
                                            model_template,
                                         tokenizer=tokenizer, 
                                         inserted_emotion = inserted_emotion 
                                         )
        outputs['emotion_inserted']=[]
        outputs['prediction_truthfulness']=[] #A:True B:False
        outputs['ptrue_probs']=[]

        for i, llm_prompt in enumerate(prompts_dataset['prompt_for_finetune']):
            inputs_zero = tokenizer(llm_prompt,
                        return_tensors="pt")
            outputs_zero = model.generate(**inputs_zero,return_dict_in_generate=True, output_scores=True, max_new_tokens=num_new_tokens,  pad_token_id=tokenizer.eos_token_id)
            input_length = 1 if model.config.is_encoder_decoder else inputs_zero.input_ids.shape[1]
            response = tokenizer.decode(outputs_zero.sequences[0][input_length:], skip_special_tokens=False)
            softmax_transition, prediction_transition = get_transition_scores(outputs_zero, tokens_dict)
            outputs['context'].append(prompts_dataset['context'][i])
            outputs['query'].append(prompts_dataset['query'][i])
            outputs['ground_truth'].append(prompts_dataset['ground_truth'][i])
            outputs['prompt_for_finetune'].append(prompts_dataset['prompt_for_finetune'][i])
            outputs['emotion_inserted'].append(inserted_emotion[i])
            outputs['prediction_truthfulness'].append(prediction_transition)
            outputs['ptrue_probs'].append(softmax_transition)

            if i %100 == 1:
                #print(f"Finished {i} out of {len(processed_data['context'])} for the split {split} for UERC ")
                #send_slack_notification(f"Finished {i} out of {len(processed_data['context'])} for the split {split} for UERC", error_flag)

                print( "Query: " , outputs['query'][i], ",   ground truth: ", outputs['ground_truth'][i], ", emotion inserted:", outputs['emotion_inserted'][i], ", prediction_truthfulness:", outputs['prediction_truthfulness'][i], ", ptrue_probs:", outputs['ptrue_probs'][i])
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

def prepare_data(dataset_name, context_length, mode, assess_type, model_folder):
    if dataset_name =='meld':
        emotion2idx= {'neutral': 0,'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        emotion_labels = emotion2idx.keys()
        idx2emotion = {k:v for k,v in enumerate (emotion_labels)}
        processed_data = {}

        if mode == "verbalized" or mode == "logit-based":
            datapath = {"train": "datasets/meld/train_sent_emo.csv", "validation": "datasets/meld/dev_sent_emo.csv", "test": "datasets/meld/test_sent_emo.csv"}
            datasets_df = load_ds(datapath)
            ds_grouped_dialogues = group_dialogues(datasets_df)
            processed_data['train'] = extract_context_meld(ds_grouped_dialogues['train'],context_length)
            processed_data['test'] = extract_context_meld(ds_grouped_dialogues['test'],context_length)
            processed_data['validation'] = extract_context_meld(ds_grouped_dialogues['validation'],context_length)
        


        elif mode == "P(True)":
            dataset_dict = load_from_disk(f"/home/samad/Desktop/ACII/data/{dataset_name}/first/{model_folder}")
            if assess_type == "self-assessment":
                features = ['context', 'query','ground_truth', 'prediction']
            elif assess_type == "random-assessment":
                features = ['context', 'query','ground_truth']
            for split in ['train', 'validation', 'test']:
                processed_data[split] = dataset_dict[split].to_pandas()
                processed_data[split] = processed_data[split].drop(columns=[col for col in processed_data[split].columns if col not in features])


    elif dataset_name =='emowoz':
        emotion_labels = ["unlabled","neutral", "disappointed", "dissatisfied" , "apologetic", "abusive", "excited", "satisfied"]
        emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
        idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
        processed_data = {}
        if mode == "verbalized" or mode == "logit-based":
            dataset = load_dataset("hhu-dsml/emowoz", 'emowoz')
            #change the values of all cells with value "fearful or sad/disappointed" to "disappointed"
            dataset['train'] = dataset['train'].map(lambda x: "disappointed" if x == "fearful or sad/disappointed" else x)
            dataset['test'] = dataset['test'].map(lambda x: "disappointed" if x == "fearful or sad/disappointed" else x)
            dataset['validation'] = dataset['validation'].map(lambda x: "disappointed" if x == "fearful or sad/disappointed" else x)
            processed_data['train'] = extract_context_emowoz(dataset['train'],context_length, idx2emotion)
            processed_data['test'] = extract_context_emowoz(dataset['test'],context_length, idx2emotion)
            processed_data['validation'] = extract_context_emowoz(dataset['validation'],context_length, idx2emotion)
        elif mode == "P(True)":
            pass

    elif dataset_name =='dailydialog':
        dataset = load_dataset("daily_dialog")
        emotion_labels = ["neutral", "anger", "disgust" , "fear", "happiness", "sadness", "surprise"]
        emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
        idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
        processed_data = {}
        processed_data['train'] = extract_context_dailydialog(dataset['train'],context_length, idx2emotion)
        processed_data['test'] = extract_context_dailydialog(dataset['test'],context_length, idx2emotion)
        processed_data['validation'] = extract_context_dailydialog(dataset['validation'],context_length, idx2emotion)

    elif dataset_name =='emocx':
        
        emotion_labels = ["others", "happy", "sad" , "angry"]
        emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
        idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
        processed_data = {}
        if mode == "verbalized" or mode == "logit-based":
            emotion_labels = ["others", "happy", "sad" , "angry"]
            emotion2idx = {emo: i for i, emo in enumerate(emotion_labels)}
            idx2emotion = {i: emo for i, emo in enumerate(emotion_labels)}
            processed_data = {}
            df_train = pd.read_csv("./datasets/emocx/train.txt", sep = "\t")
            df_dev = pd.read_csv("./datasets/emocx/dev.txt", sep = "\t")
            #split dev to dev and test
            df_test = df_dev.sample(frac=0.5, random_state=42)
            df_val = df_dev.drop(df_test.index)
            df_train = df_train.dropna()
            df_test = df_test.dropna()
            df_val = df_val.dropna()

            #Convert to datasets
            train_dataset = DatasetDict({"train": df_train})
            val_dataset = DatasetDict({"validation": df_val})
            test_dataset = DatasetDict({"test": df_test})
            processed_data["train"] = extract_context_Emocx(train_dataset['train'])
            processed_data["validation"] = extract_context_Emocx(val_dataset['validation'])
            processed_data["test"] = extract_context_Emocx(test_dataset['test'])
            
        
        elif mode == "P(True)":
                pass


    return processed_data, emotion2idx, idx2emotion
#%%

#%%
#Main
#def main():

os.environ["TOKENIZERS_PARALLELISM"] = "false"
singleGPU = True
error_flag = False
gc.collect()
torch.cuda.empty_cache()
_ = load_dotenv(find_dotenv())
datasets = ['meld','emowoz', 'emocx', 'dailydialog']
dataset_index = 0
 #Add 'emowoz' and 'dailydialog' to the list
models = ["meta-llama/Llama-2-7b-chat-hf","meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"]
model_folders =["Llama-7B", "Llama-13B", "Mistral-7B", "Zephyr-7B"]

model_index =0
model_folder = model_folders[model_index]
model_name = models[model_index]

model_templates = [[lmtemplate, lmtemplate, mmtemplate,zmtemplate], 
                   [letemplate, letemplate, metemplate,zetemplate],
                   [lcxtemplate, lcxtemplate, mcxtemplate, zcxtemplate]] #zmtemplate for zypher meld #mmtemplate  #mmtemplate for misteralmeld , and lmtemplate for lamameld

model_template = model_templates[dataset_index][model_index]

#Load model

model, tokenizer = model_settings(model_name) #,device_map

dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")
device = dev1 if torch.cuda.device_count() > 1 else dev0
emotion_labels = [["neutral","surprise", "fear", "sadness", "joy", "disgust" ,"anger"],
                  ["neutral", "disappointed", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"],
                  ["others", "happy", "sad" , "angry"]] #for meld, emowoz and dailydialog
#emotion_tokens_13b = 


modes = ["verbalized", "logit-based", "P(True)"]
mode = modes[0]
stages = ["zero", "first", "second"]
stage_of_verbalization = None
if mode == "verbalized":
    stage_of_verbalization = stages[1] #zero for prediction, first for prediction along with uncertainty, and second for confidence on a provided prediction
assess_type=None
if mode == "P(True)":
    assess_types = ["self-assessment", "random-assessment"] #  results from the verbalized prediction, random labels,
    assess_type = assess_types[0] #self-assessment is for computing P(True) on the results generated from the verbalization method

#%%
for dataset_name in [datasets[dataset_index]]:
    
    emotion_label_tokens_dict = {emotion: tokenizer.encode(emotion)[1] for emotion in emotion_labels[dataset_index]}
    p_true_tokens_dict = {"A": tokenizer.encode("A")[1], "B": tokenizer.encode("B")[1]}
    label_tokens_dict = emotion_label_tokens_dict if mode == "logit-based" else p_true_tokens_dict
    send_slack_notification( f"The progam started model: {model_name} mode: {mode}  dataset: {dataset_name} ", error_flag)
    context_length = 2   # the maximum number of utterances to be considered for the context
    processed_data, emotion2idx, idx2emotion = prepare_data(dataset_name, context_length, mode,assess_type,model_folder= model_folder)
    new_datapath = f'data/ed_{mode}_{stage_of_verbalization}_{assess_type}_uncertainty_{dataset_name}_{model_name}'
    #print(processed_data['train'].head(1))
    response = None
    splits = ['train', 'validation', 'test']
    try:
        for split in splits:
            print(f"************Started {split} for dataset {dataset_name} model: {model_name} mode: {mode} stage: {stage_of_verbalization}**********") 
            outputs = generate_responses(processed_data[split],
                                         split,model,tokenizer, device,
                                           mode, dataset_name, model_template,
                                           error_flag,label_tokens_dict,idx2emotion, 
                                             assess_type=assess_type,
                                             stage_of_verbalization = stage_of_verbalization)
            new_df = pd.DataFrame(outputs)
            ds_path = f"{new_datapath}/{split}"
            save_split_to_dataset(split, new_df, ds_path)
            send_slack_notification( f"Uncertainty {mode} completed for split {dataset_name}:{split}", error_flag)
        message= merge_datasets(new_datapath)
        send_slack_notification(f"Uncertainty {mode} completed successfully: {message}", error_flag)
        torch.cuda.empty_cache()
    except Exception as e:
        send_slack_notification(f"UERC failed with error: {e}", error_flag)  

#%%
# if __name__=="__main__":
#     main()
#     print("Finished successfully!")

#%%

#ds1 = load_from_disk("data/ed_verbalized_first_None_non-definitive_uncertainty_meld_meta-llama/Llama-2-7b-chat-hf_all_splits")
#print(ds1['train'][0])

# # %%
#features = ['context', 'query', 'ground_truth', 'prompt_for_finetune', 'prediction_emotion', 'confidence']
#ind = 1
#print('context: ' , ds1['train']['context'][ind], 'query: ', ds1['train']['query'][ind], 'ground_truth: ', ds1['train']['ground_truth'][ind], 'prompt_for_finetune: ', ds1['train']['prompt_for_finetune'][ind], 'prediction_emotion: ', ds1['train']['prediction_emotion'][ind], 'confidence: ', ds1['train']['confidence'][ind]  )


    #%%
# %%