B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

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
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
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

Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        


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


Here is another example of how an uncertainty-aware emotion recognition in conversation assistant should work:


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

####""" + E_SYS+ f"""\n Remember that you always provide the output in a JSON format, with emotion labels as keys and confidences as values, without any extra explanation.

Remember that emotion labels you should provide confidence values in the JSON file for all the labels in the folloqing list: neutral, surprise, fear, sadness, joy, disgust, anger.

 Highest confidence belongs to the most likely emotion, and the sum of all confidence values should be exactly 1.0.

Cosidering the context following, what is your confidence values for the following query utterance?
 
     context: {context} 

    query utterance: {query}


""" + E_INST+ "Output JSON string:\n" 

    elif stage_of_verbalization  == "zero":

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
        
    elif stage_of_verbalization == "first":
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    You have two following tasks:
     
    First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    "neutral" 
    "surprise" 
    "fear" 
    "sadness" 
    "joy" 
    "disgust" 
    "anger"


If the query utterance does not carry any clear emotion, the output is: [neutral]

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

####
Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        
    
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


####""" + E_SYS+ f"""Remember that you always provide your prediction from the list of emotion labels: [neutral, surprise, fear, sadness, joy, disgust, anger], and confidence on that preduction in JSON string format.

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.

What is your prediction and confidence on that prediction for the following query utterance?


    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output JSON string:\n" 
        
    elif stage_of_verbalization == "second":
        potential_labels = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        if exclude_label !='neutral' and exclude_label!=None:
            potential_labels = [label for label in potential_labels if label != exclude_label ]
        prompt= "<s>" + B_INST +B_SYS+ f""" You are helpful, respectful and honest emotion recognition in conversation assistant. 
    You always will be given a potential labels of emotion, delimited by triple backticks, a context of conversation and a query utterance. Your task is to analyze the context of the given 
    conversation and predict which label from the given potential list best conveys the emotional state of the query utterance. 


If the query utterance does not convey a clear emotion, you should choose [neutral]. 


Here is an example of how an emotion recognition in conversation assistant should work:        

####
---Input:
    
    In the following conversation, considering the provided context and the emotions list ```['neutral', 'surprise', 'fear', 'sadness']```, which label best reflects the emotional state of the query utterance?
    
    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
---Output:

    the correct label is: sadness


Here is another example of how an emotion recognition in conversation assistant should work:

---Input:

    In the following conversation, considering the provided context and the emotions list ```['neutral', 'surprise', 'fear', 'joy']```, which label best reflects the emotional state of the query utterance?
    
    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

    
---Output:

    The correct label is: neutral
    
    ####""" + E_SYS+ f"""

    Remember that you are helpful, respectful and honest emotion recognition in conversation assistant.

    Remember that your never response with a label that is not included in the following potential emotion labels list. 

    In the following conversation, considering the context, which label from potential emotion list ```{potential_labels}``` best reflects the emotional state of the  query utterance.
    
---Input

    context: {context} 

    query utterance: {query}

    potential emotion labels: ```{potential_labels}```


---Output:

""" + E_INST+ "\tThe correct label is: " 
    


    return prompt



def meld_logit(context, query,tokenizer,emotion_label):
     prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    neutral 
    surprise 
    fear 
    sadness 
    joy 
    disgust 
    anger


If the query utterance does not carry any clear emotion, the output is: neutral

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 


####

Here is an example of how an emotion recognition in conversation assistant should work:        


    context: [Monica]: You never knew she was a lesbian? [surprise]
            [Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]
    
    query utterance: [Monica]: I am sorry

    
Output string: sadness


Here is another example of how an emotion recognition in conversation assistant should work:


    context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

    query utterance: [Chandler]: That I did. That I did.

Output string: neutral

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
 
 what is your prediction for the following query utterance?
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
     return prompt

def meld_ptrue(context, query, tokenizer,emotion_label):

    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context and query utterance of a conversation and determine if: 

    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label than the proposed label.

    
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

---Input:

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
####

""" + E_SYS+ f""" Remember that you are a helpful, respectful and honest emotion recognition in conversation assistant and your task is to carefully analyze the context and query utterance of a conversation and determine if: 
    
    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label from the give motional states list than the proposed label.


Here is a new conversation:

---Input:

    Context: {context}
        
    Query utterance: {query} 

Considering the provided context and the emotions list ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'], would ```{emotion_label}``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No


---Output:
 
 """ + E_INST +"The correct answer is: "
    return prompt
























































