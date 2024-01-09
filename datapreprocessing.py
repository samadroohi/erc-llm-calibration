#%%
from datasets import load_dataset
from datasets import DatasetDict, Dataset
import pandas as pd

#%%
def load_ds(path):
    train_dataset = load_dataset('csv', data_files=path['train'], split='train')
    test_dataset = load_dataset('csv', data_files=path['test'], split='train')
    val_dataset = load_dataset('csv', data_files=path['validation'], split='train')
    train_valid_test = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
    train_valid_test.set_format(type ="pandas")
    df_train = train_valid_test["train"][:]
    df_test = train_valid_test["test"][:]
    df_val = train_valid_test["validation"][:]
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_val = df_val.dropna()
    df_train = df_train.drop(df_train[df_train['Emotion'] == 'Emotion'].index)
    df_test = df_test.drop(df_test[df_test['Emotion'] == 'Emotion'].index)
    df_val = df_val.drop(df_val[df_val['Emotion'] == 'Emotion'].index)
    return {"train": df_train, "validation": df_val, "test": df_test}
def group_dialogues(dataset_df):
    dialogues={}
    for ds in dataset_df:
        ds_dialogues=[]
        grouped_dialogues = dataset_df[ds].groupby('Dialogue_ID')
        for name, group in grouped_dialogues:
            ds_dialogues.append(group)
        dialogues[ds] = ds_dialogues
    return dialogues
    
def format_contex(context):
    context_fr =''
    for _, ctx in context.iterrows():
        #print(ctx)
        context_fr += f'[{ctx["Speaker"]}]: {ctx["Utterance"]} [{ctx["Emotion"]}]'

    return context_fr
def extract_context_meld(groupped_df, context_window, emotion2idx):
    data_frame = pd.DataFrame()
    for dialogue in groupped_df: 
        for i in range(0, len(dialogue)-context_window-1):
            context = dialogue[i:i+context_window]
            context_fr =format_contex(context)
            #print(context_fr)
            query_utterance = dialogue.iloc[i+context_window]['Utterance']
            query_emotion = emotion2idx[dialogue.iloc[i+context_window]['Emotion']]
            query_speaker = dialogue.iloc[i+context_window]['Speaker']
            query = f'[{query_speaker}]:{query_utterance}'
            dlg = pd.DataFrame({'context':context_fr,'query': query, 'emotion':query_emotion}, index=[i])
            data_frame = pd.concat([data_frame, dlg], ignore_index=True)
    return data_frame
def extract_context_emowoz(dataset, context_length, idx2emotion):
    data = {"context":[],	"query":[], "emotion":[]}
    turns = ['human', 'agent']
    for i , dlg in enumerate(dataset['log']):
        for j in range(0,len(dlg['text'])-context_length,context_length):
            context_human = f"[{turns[0]}]: {dlg['text'][j]}[{idx2emotion[dlg['emotion'][j]+1]}]"
            context_agent = f"[{turns[1]}]: {dlg['text'][j+1]}[{idx2emotion[dlg['emotion'][j+1]+1]}]"
            context = context_human +' , '+ context_agent
            query = f"[{turns[0]}]: {dlg['text'][j+2]}"
            data['context'].append(context)
            data['query'].append(query)
            data['emotion'].append(dlg['emotion'][j+2])
    data = pd.DataFrame(data)   
    return data

def extract_context_dailydialog(dataset, context_length, idx2emotion):
    data = {"context":[],	"query":[], "emotion":[]}
    turns = ['speaker1', 'speaker2']
    for i , dlg in enumerate(zip(dataset['dialog'], dataset['emotion'])):
        for j in range(0,len(dlg[0])-context_length,context_length):
            context_speaker1 = f"[{turns[0]}]: {dlg[0][j]}[{idx2emotion[dlg[1][j]]}]"
            context_speaker2= f"[{turns[1]}]: {dlg[0][j+1]}[{idx2emotion[dlg[1][j+1]]}]"
            context = context_speaker1 +' , '+ context_speaker2
            query = f"[{turns[0]}]: {dlg[0][j+2]}"
            data['context'].append(context)
            data['query'].append(query)
            data['emotion'].append(dlg[1][j+2])
    data = pd.DataFrame(data)   
    return data

# %%
# def main():
#     #hyperparameters
#     start_contexting = 1 # the number of dialogues to be considered for the context
#     context_length = 2 # the maximum number of utterances to be considered for the context
#     datapath = {"train": "datasets/meld/train_sent_emo.csv", "validation": "datasets/meld/dev_sent_emo.csv", "test": "datasets/meld/test_sent_emo.csv"}
#     datasets_df = load_ds(datapath)
#     emotion_labels = datasets_df['train']['Emotion'].unique().tolist()
#     emotion2idx = {v:k for k,v in enumerate (emotion_labels)}
#     idx2emotion = {k:v for k,v in enumerate (emotion_labels)}
#     print(f'Emotion to index: {emotion2idx}')
#     print(f'Index to emotion: {idx2emotion}')
#     ds_grouped_dialogues = group_dialogues(datasets_df)
#     train_proccessed = extract_context(ds_grouped_dialogues['train'],context_length)
#     test_proccessed = extract_context(ds_grouped_dialogues['test'],context_length)
#     val_proccessed = extract_context(ds_grouped_dialogues['validation'],context_length)
#     print(train_proccessed.head(1))
    

# if __name__ == "__main__":
#     main()
# # %%
