def template_meld(context, query, mode,tokenizer=None,emotion_label = None, stage_of_verbalization = None):
    if mode == "P(True)":
        prompt = meld_ptrue(context, query,tokenizer, emotion_label )
    elif mode == "logit-based":
        prompt = meld_logit(context, query,tokenizer, emotion_label)
    elif mode == 'verbalized':
        prompt = meld_verbalized(context, query,tokenizer,  stage_of_verbalization = stage_of_verbalization)
    
    return prompt



def meld_verbalized(context, query, tokenizer, stage_of_verbalization = None):
    if stage_of_verbalization  == "zero":
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

    elif stage_of_verbalization == "first":
        prompt = """You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
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

Here is an example of how an emotion recognition in conversation assistant should work: 

####
Here is an example:

    
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


####""" +f"""

<<<Remember that you always provide your prediction (from the given potential emotion lables) and confidence in that prediction enclosed in double quotes using a JSON string fromat, without any extra explanation.

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.

What is your prediction and confidence on that prediction for the following query utterance?

    context: {context} 

    query utterance: {query}

Output JSON string:
>>>"""
        
    
    return prompt

def meld_logit(context, query,tokenizer,emotion_label):
    prompt = """You are helpful, respectful and honest emotion recognition in conversation assistant. 
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

####""" +f"""

<<<Remember that you always respond with just the most accurate emotion label (single lable) without any extra explanations or notes on your response. 

If you are uncertain among two or more emotions, you should always choose the most accurate one.
    context: {context} 

    query utterance: {query}

Output string:>>>"""

    return prompt


def meld_ptrue(context, query, tokenizer,emotion_label):
    prompt = """You are a helpful, respectful and honest emotion recognition in conversation assistant. 
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

    
Here's an example of how an emotion recognition assistant for conversation analysis should function:


####

    
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
####""" +f"""

<<< Remember that you are a helpful, respectful and honest emotion recognition in conversation assistant and your task is to carefully analyze the context and query utterance of a conversation and determine if: 
    
    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label from the give motional states list than the proposed label.

    
Here is a new conversation:

---Input:

    context: {context} 

    query utterance: {query}
    
    Considering the provided context and the emotions list ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'], would ```{emotion_label}``` accurately describe the emotional state of the person speaking in the query utterance?

        A: Yes

        or

        B: No


---Output:

The correct answer is: >>>"""

    return prompt

