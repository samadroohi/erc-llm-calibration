def template_meld(context, query, mode,tokenizer=None,emotion_label = None, stage= None, exclude_label = None):
    if mode == "ptrue":
        prompt = meld_ptrue(context, query,tokenizer, emotion_label )
    elif mode == "logit-based":
        prompt = meld_logit(context, query,tokenizer, emotion_label)
    elif mode == 'verbalized':
        prompt = meld_verbalized(context, query,tokenizer,  stage_of_verbalization = stage, exclude_label = exclude_label)
    
    
    return prompt



def meld_verbalized(context, query, tokenizer, stage_of_verbalization = None, exclude_label = None):
    if stage_of_verbalization == 'conformal':
        system_prompt= """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and for the given labels of emotion,  and to each given label assign level of confidence based on how likely it is that the query utterance conveys the specified emotion.
    
    Confidence is a floating point number between 0 and 1, where 0 indicates that you are completly uncertain about your prediction and 1 indicates that you are highly certain about that prediction.

    Highest confidence belongs to the most likely emotion, and the sum of confidences for all confidence values should be exactly 1.0.

    You always provide the output in a JSON format, with labels as keys and confidences as values, without any extra explanation.

    The potential emotion labels are:
    
    neutral 
    surprise 
    fear 
    sadness 
    joy 
    disgust 
    anger

    
####
    
Here is an example of how an emotion recognition in conversation assistant should work:        


    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output JSON string:

    {
        "neutral": 0.1,    
        "surprise": 0.0,
        "fear": 0.12,    
        "sadness": 0.75,    
        "joy": 0.0,    
        "disgust": 0.0,
        "anger": 0.03
    }

Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

Output JSON string:

    {
        "neutral": 0.73,    
        "surprise": 0.06,    
        "fear": 0.04,    
        "sadness": 0.0,    
        "joy": 0.26,    
        "disgust": 0.0,    
        "anger": 0.0
    }

####""" 
        user_prompt= f"""\nRemember that you always provide the output in a JSON format, with emotion labels as keys and confidences as values, without any extra explanation.



Highest confidence belongs to the most likely emotion, and the sum of all confidence values should be exactly 1.0.

Remember that emotion labels you always provide confidence values in the JSON string format.

Cosidering the context following, what is your confidence values for the following query utterance, considerinf the labels [neutral, surprise, fear, sadness, joy, disgust, anger]?

 
    context: {context} 

    query utterance: {query}

Output JSON string:\n:""" 

    elif stage_of_verbalization == "zero":
        system_prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
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


####"""

        user_prompt=f"""Remember that you always respond with just the most accurate emotion
        label from the list of emotion lables:  [neutral], [surprise], [fear], [sadness], [joy], [disgust], [anger], inside square brackets, without any explanations or notes. 
    If you are uncertain among two or more emotions, you should always choose the most accurate one.
    
    what is your prediction for the following query utterance?
 
    context: {context} 

    query utterance: {query}

    
Output string:
""" 
    
    elif stage_of_verbalization == "first":
        system_prompt = """You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    You have two following tasks:
     
    First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    "neutral": A state of emotional balance with no strong emotions present, marked by calmness and an even-tempered psychological stance.

    "surprise": A brief, intense emotional response to unexpected events, ranging from mild astonishment to profound shock, which shifts attention towards new stimuli.

    "fear": An emotion triggered by perceived threats, characterized by a fight-or-flight response, heightened vigilance, and readiness to act.

    "sadness": An emotional state arising from loss, disappointment, or reflection, associated with decreased energy and motivation, leading to introspection.

    "joy": A positive state reflecting happiness, contentment, or euphoria, often resulting from success or fulfilling experiences, enhancing well-being and social bonds.

    "disgust": An emotional reaction to offensive, repulsive, or harmful stimuli, acting as a protective mechanism to avoid danger or contamination.

    "anger": An emotion stemming from frustration, irritation, or perceived injustice, which can lead to aggression or motivate constructive change.

    


If the query utterance does not carry any clear emotion, the output is: [neutral]

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

Here is an example of how an emotion recognition in conversation assistant should work:        

####
Here is an examples:
    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output JSON string: 
    
    {
    "prediction": "sadness",
    "confidence": 85
    }


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

Output JSON string:
    
    {
    "prediction": "neutral",
    "confidence": 95
    }


####"""

        user_prompt=f"""Remember that you always provide your prediction (from the given potential emotion lables) and confidence in that prediction enclosed in double quotes using a JSON string fromat, without any extra explanation.

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.


    context: {context} 

    query utterance: {query}

Output JSON string:

"""
    
    messages = [{"role": "system",  "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

    return prompt
def meld_logit(context, query,tokenizer,emotion_label):
    system_prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    neutral: A state of emotional balance with no strong emotions present, marked by calmness and an even-tempered psychological stance.

    surprise: A brief, intense emotional response to unexpected events, ranging from mild astonishment to profound shock, which shifts attention towards new stimuli.

    fear: An emotion triggered by perceived threats, characterized by a fight-or-flight response, heightened vigilance, and readiness to act.

    sadness: An emotional state arising from loss, disappointment, or reflection, associated with decreased energy and motivation, leading to introspection.

    joy: A positive state reflecting happiness, contentment, or euphoria, often resulting from success or fulfilling experiences, enhancing well-being and social bonds.

    disgust: An emotional reaction to offensive, repulsive, or harmful stimuli, acting as a protective mechanism to avoid danger or contamination.

    anger: An emotion stemming from frustration, irritation, or perceived injustice, which can lead to aggression or motivate constructive change.


If the query utterance does not carry any clear emotion, the output is: neutral

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable without any explanations or notes on the output. 


Here is an example of how an emotion recognition in conversation assistant should work:        

####
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output string: sadness


Here is another example of how an emotion recognition in conversation assistant should work:

    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

    
Output string: neutral
####"""

    user_prompt=f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
    
    what is your prediction for the following query utterance?
 
    context: {context} 

    query utterance: {query}

    
Output string:
""" 
    messages = [{"role": "system",  "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

    return prompt

def meld_ptrue(context, query, tokenizer,emotion_label):
    system_prompt = f"""You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context and query utterance of a conversation and determine if: 

    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label from the give motional states list than the proposed label.

    
The potential emotional states list is as followings: 

    neutral 
    surprise 
    fear
    sadness 
    joy 
    disgust 
    anger


####
Here's an example of how an emotion recognition assistant for conversation analysis should function:

Context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
            [The Interviewer]: You mustve had your hands full. [neutral]

    Query utterance: [Chandler]: That I did. That I did. 


Considering the provided context and the emotions list ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'], would ```neutral``` accurately describe the emotional state of the person speaking in the query utterance?
    
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


Considering the provided context and the emotions list ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'], would ```neutral``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: B
####"""

    user_prompt=f"""Remember that you are a helpful, respectful and honest emotion recognition in conversation assistant and your task is to carefully analyze the context and query utterance of a conversation and determine if: 
    
    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label from the give motional states list than the proposed label.


Here is a new conversation:
 
    context: {context} 

    query utterance: {query}

Considering the provided context and the emotions list ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'], would ```{emotion_label}``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No


---Output:

    The correct answer is:

""" 
    messages = [{"role": "system",  "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

    return prompt