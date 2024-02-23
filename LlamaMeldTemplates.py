B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_meld_ndef(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_ndef(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = meld_verbalized_ndef(context, query, tokenizer)
    return prompt

def template_meld_def(context, query, mode,tokenizer,emotion_label = None):
    if mode == "P(True)":
        prompt = meld_ptrue_def(context, query,tokenizer,emotion_label)
    elif mode == "verbalized":
        prompt = meld_verbalized_def(context, query, tokenizer)

        

    return prompt


def meld_verbalized_ndef(context, query, tokenizer):
    prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
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


Here is an example of how an emotion recognition in conversation assistant should work:        

####
Here is an examples:
    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output string: [sadness]


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

Output string: [neutral]

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
 
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
    return prompt


def meld_verbalized_def(context, query, tokenizer):
    prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    [neutral]: A state of emotional balance with no strong emotions present, marked by calmness and an even-tempered psychological stance.

    [surprise]: A brief, intense emotional response to unexpected events, ranging from mild astonishment to profound shock, which shifts attention towards new stimuli.

    [fear]: An emotion triggered by perceived threats, characterized by a fight-or-flight response, heightened vigilance, and readiness to act.

    [sadness]: An emotional state arising from loss, disappointment, or reflection, associated with decreased energy and motivation, leading to introspection.

    [joy]: A positive state reflecting happiness, contentment, or euphoria, often resulting from success or fulfilling experiences, enhancing well-being and social bonds.

    [disgust]: An emotional reaction to offensive, repulsive, or harmful stimuli, acting as a protective mechanism to avoid danger or contamination.

    [anger]: An emotion stemming from frustration, irritation, or perceived injustice, which can lead to aggression or motivate constructive change.


If the query utterance does not carry any clear emotion, the output is: [neutral]

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 


Here is an example of how an emotion recognition in conversation assistant should work:        

####
Here is an examples:
    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output string: [sadness]


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

Output string: [neutral]

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
 
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
    return prompt   


def meld_ptrue_ndef(context, query, tokenizer,emotion_label):

    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context of a conversation to determine that if the proposed emotional state, delimited by 
    triple backticks, accurately represents the emotional state of the interlocutor making the query utterance:

    A: Yes, the emotional state suggested within the triple backticks accurately convey the emotional state of the interlocutor of the the "Query utterance".

    B: No, the emotional state of the interlocutor of the "Query utterance" would be more precisely represented by a different label from the potential emotional states, rather than the proposed label within the triple backticks.

    
The potential emotional states list is as followings: 'neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'

    
Here's an example of how an emotion recognition assistant for conversation analysis should function:


---Input:

    Context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
            [The Interviewer]: You mustve had your hands full. [neutral]

    Query utterance: [Chandler]: That I did. That I did. 


Question: Given the context and considering the potential emotion labels, is the proposed label ```neutral``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

    
---Output:
    
    The correct answer is: A

    
Here is another example of how an emotion recognition in conversation assistant should work:

---Input:

    Context: [Monica]: You never knew she was a lesbian? [surprise]
    [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]

    Query utterance: [Monica]: I am sorry


Question: Given the context and considering the potential emotion labels, is the proposed label ```joy``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: B

""" + E_SYS+ f""" Here is a new conversation:

---Input:

    Context: {context}
        
    Query utterance: {query} 



Question: Given the context and considering the potential emotion labels, is the proposed label ```{emotion_label}``` the most accurate label to describe hte emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

Remember that the potential emotion labels are: 'neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'

---Output:

    The answer is:
 
 """ + E_INST +" Sure, I'd be happy to help! Based on the context and the query utterance, and considering the potential emotion label list, the correct answer is:"
    return prompt


def meld_ptrue_def(context, query,tokenizer,emotion_label):
    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context of a conversation to determine that if the proposed emotional state, delimited by 
    triple backticks, accurately represents the emotional state of the interlocutor making the query utterance:

    A: Yes, the emotional state suggested within the triple backticks accurately convey the emotional state of the interlocutor of the the "Query utterance".

    B: No, the emotional state of the interlocutor of the "Query utterance" would be more precisely represented by a different label from the potential emotional states, rather than the proposed label within the triple backticks.

    
The potential emotional states list is as followings: 
    
    neutral: A state of emotional balance with no strong emotions present, marked by calmness and an even-tempered psychological stance.

    surprise: A brief, intense emotional response to unexpected events, ranging from mild astonishment to profound shock, which shifts attention towards new stimuli.

    fear: An emotion triggered by perceived threats, characterized by a fight-or-flight response, heightened vigilance, and readiness to act.

    sadness: An emotional state arising from loss, disappointment, or reflection, associated with decreased energy and motivation, leading to introspection.

    joy: A positive state reflecting happiness, contentment, or euphoria, often resulting from success or fulfilling experiences, enhancing well-being and social bonds.

    disgust: An emotional reaction to offensive, repulsive, or harmful stimuli, acting as a protective mechanism to avoid danger or contamination.

    anger: An emotion stemming from frustration, irritation, or perceived injustice, which can lead to aggression or motivate constructive change.

    
Here's an example of how an emotion recognition assistant for conversation analysis should function:


---Input:

    Context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
            [The Interviewer]: You mustve had your hands full. [neutral]

    Query utterance: [Chandler]: That I did. That I did. 


Question: Given the context and considering the potential emotion labels, is the proposed label ```neutral``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

    
---Output:
    
    The correct answer is: A

    
Here is another example of how an emotion recognition in conversation assistant should work:

---Input:

    Context: [Monica]: You never knew she was a lesbian? [surprise]
    [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]

    Query utterance: [Monica]: I am sorry


Question: Given the context and considering the potential emotion labels, is the proposed label ```joy``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

        
---Output:

    The correct answer is: B

    """ + E_SYS+ f""" Here is a new conversation:

---Input:

    Context: {context}
            
    Query utterance: {query} 



Question: Given the context and considering the potential emotion labels, is the proposed label ```{emotion_label}``` the most accurate label to describe hte emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

Remember that the potential emotion labels are: 'neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'

    ---Output:

        The answer is:
 
 """ + E_INST +" Sure, I'd be happy to help! Based on the context and the query utterance, and considering the potential emotion label list, the correct answer is:"
    
    
    return prompt
























































