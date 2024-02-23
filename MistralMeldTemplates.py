def template_meld_ndef(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_ndef(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = meld_verbalized_ndef(context,query, tokenizer)
    return prompt

def template_meld_def(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_def(context, query,tokenizer,emotion_label)
    elif mode == "verbalized":
        prompt = meld_verbalized_def(context, query, tokenizer)

    return prompt

def meld_verbalized_ndef(context, query, tokenizer):
    prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    [neutral] 
    [surprise] 
    [fear] 
    [sadness] 
    [joy] 
    [disgust] 
    [anger]


If the Query utterance does not carry any clear emotion, the output is: [neutral]

You always just output the accurate emotional state of the <<<Query utterance>>> without any explanation. 

You will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.


####
Here are some examples:

    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Category: [sadness]


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.


Category: [neutral]

Remember that you will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.

####
<<<
    context: {context} 

    query utterance: {query}


Category:>>>""" 
    return prompt

def meld_verbalized_def(context, query, tokenizer):
    pass

def meld_ptrue_ndef(context, query, tokenizer,emotion_label):
    pass

def meld_ptrue_def(context, query,tokenizer, emotion_label):
    pass