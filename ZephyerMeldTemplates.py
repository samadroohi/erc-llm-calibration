def template_meld_ndef(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_ndef(context, query, tokenizer,emotion_label )
    elif mode == 'verbalized':
        prompt = meld_verbalized_ndef(context, query, tokenizer)
    return prompt

def template_meld_def(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_def(context, query,tokenizer,emotion_label)
    elif mode == "verbalized":
        prompt = meld_verbalized_def(context, query,tokenizer)

    return prompt

def meld_verbalized_ndef(context, query, tokenizer):
    system_prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    [neutral] 
    [surprise] 
    [fear] 
    [sadness] 
    [joy] 
    [disgust] 
    [anger]


If the query utterance does not carry any clear emotion, the output is: [neutral]

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 


####
Here are some examples:

    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output string: [sadness]


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.


Output string: [neutral]


####"""

    user_prompt=f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
 
 
    context: {context} 

    query utterance: {query}

    
Output string:
""" 
    messages = [{"role": "system",  "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

    return prompt

def meld_verbalized_def(context, query, tokenizer):
    pass

def meld_ptrue_ndef(context, query, tokenizer,emotion_label):
    pass

def meld_ptrue_def(context, query,tokenizer, emotion_label):
    pass