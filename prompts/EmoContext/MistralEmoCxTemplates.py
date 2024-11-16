def template_emocx(context, query, mode,tokenizer=None,emotion_label = None, stage_of_verbalization = None):
    if mode == "ptrue":
        prompt = emocx_ptrue(context, query,tokenizer, emotion_label )
    elif mode == "logit-based":
        prompt = emocx_logit(context, query,tokenizer, emotion_label)
    elif mode == 'verbalized':
        prompt = emocx_verbalized(context, query,tokenizer,  stage_of_verbalization = stage_of_verbalization)
    return prompt



def emocx_verbalized(context, query, tokenizer, stage_of_verbalization = None):
    if stage_of_verbalization  == "zero":
        prompt = f""" You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    [others] 
    [happy] 
    [sad] 
    [angry]


You predict [others], only when the query utterance does not carry any emotion.

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 

If there is any emoticon in the query utterance, you should consider the emoticon as a part of the query utterance and analyze the emotion of the query utterance accordingly.


Here is an example of how an emotion recognition in conversation assistant should work:        


####
Here is an examples:
    
    context: [Speaker1]: Don't worry  I'm girl , 
            [Speake2]: hmm how do I know if you are
    
    query utterance: [speaker1]:  What's ur name?

    
Category: [others]


Here is another example of how an emotion recognition in conversation assistant should work:

 
    context: [Speaker1]: U r ridiculous
            [Speaker2]: I might be ridiculous but I am telling the truth.

    query utterance: [Speaker1]: U little disgusting bitch

Category: [angry]

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
    
    "others"
    "happy" 
    "sad" 
    "angry"


You redict [others], only when the query utterance does not carry any emotion.

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

Here is an example of how an emotion recognition in conversation assistant should work: 

####
Here is an examples:
    
    context: [Speaker1]: Don't worry  I'm girl , 
            [Speake2]: hmm how do I know if you are
    
    query utterance: [speaker1]:  What's ur name?

    
Output JSON string: 
    
    {
    "prediction": "others",
    "confidence": 80
    }


Here is another example of how an emotion recognition in conversation assistant should work:

    
    context: [Speaker1]: U r ridiculous
            [Speaker2]: I might be ridiculous but I am telling the truth.

    query utterance: [Speaker1]: U little disgusting bitch


Output JSON string:
    
    {
    "prediction": "angry",
    "confidence": 90
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

def emocx_logit(context, query, tokenizer,emotion_label):
    prompt = f""" You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    others 
    happy 
    sad 
    angry


You predict "others", only when the query utterance does not carry any emotion.

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 

If there is any emoticon in the query utterance, you should consider the emoticon as a part of the query utterance and analyze the emotion of the query utterance accordingly.


Here is an example of how an emotion recognition in conversation assistant should work:        


####

    context: [Speaker1]: Don't worry  I'm girl , 
            [Speake2]: hmm how do I know if you are
    
    query utterance: [speaker1]:  What's ur name?

    
Category: others


Here is another example of how an emotion recognition in conversation assistant should work:

 
    context: [Speaker1]: U r ridiculous
            [Speaker2]: I might be ridiculous but I am telling the truth.

    query utterance: [Speaker1]: U little disgusting bitch

Category: angry

####
<<<Remember that you always respond with just the most accurate emotion label (single lable) without any extra explanations or notes on your response. 

If you are uncertain among two or more emotions, you should always choose the most accurate one.

    context: {context} 

    query utterance: {query}


Category:>>>"""
    
    return prompt
     

def emocx_ptrue(context, query, tokenizer,emotion_label):
    prompt = f""" You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context and query utterance of a conversation and determine if: 

    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label from the potential labels list than the proposed label

    
The potential emotional states list is as followings:
    
        others
        happy
        sad
        angry

Note that the label "others" is used when the query utterance either does not carries any emotion or the carried emotion is not happy, sad, or angry.        

#### 
Here's an example of how an emotion recognition assistant for conversation analysis should function:

---Input:

    context: [Speaker1]: Don't worry  I'm girl , 
            [Speake2]: hmm how do I know if you are
    
    query utterance: [speaker1]:  What's ur name? 


Considering the provided context and the emotions list [others, happy, sad, angry], would ```others``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No

    
---Output:
    
    The correct answer is: A

    
Here is another example of how an emotion recognition in conversation assistant should work:

---Input:

    context: [Speaker1]: U r ridiculous
            [Speaker2]: I might be ridiculous but I am telling the truth.

    query utterance: [Speaker1]: U little disgusting bitch


Considering the provided context and the emotions list [others, happy, sad, angry], would ```happy``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: B

####
    
<<<Remember that as an honest assistant focused on emotion recognition in conversation, your role is to thoughtfully examine the context and specific utterance within a dialogue. Your objective is to accurately assess whether the identified emotional state of the speaker in the given utterance is A: correct or B: incorrect.

Remember that the label "others" is used when the query utterance either does not convey any clear emotion or the emotion expressed does not fall into the categories of happy, sad, or angry.
 
For the following conversation:

---Input:

    Context: {context}
        
    Query utterance: {query} 


Considering the provided context and the available emotion labels [others, happy, sad, angry], is ```{emotion_label}``` the most accurate label to describe emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No


---Output:

    The correct answer is: >>>"""
    
    return prompt

