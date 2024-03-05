B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_emocx(context, query, mode,tokenizer=None,emotion_label = None, stage_of_verbalization = None):
    if mode == "P(True)":
        prompt = emocx_ptrue(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = emocx_verbalized(context, query,tokenizer,  stage_of_verbalization = stage_of_verbalization)
    return prompt



def emocx_verbalized(context, query, tokenizer, stage_of_verbalization = None):
    if stage_of_verbalization  == "zero":

        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    [others] 
    [happy] 
    [sad] 
    [angry]


You redict [others], only when the query utterance does not carry any emotion.

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

    
Output string: [others]


Here is another example of how an emotion recognition in conversation assistant should work:

 
    context: [Speaker1]: U r ridiculous
            [Speaker2]: I might be ridiculous but I am telling the truth.

    query utterance: [Speaker1]: U little disgusting bitch

Output string: [angry]

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. 

Remember that you predict [others], only when the query utterance does not carry any emotion.
 

    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
        
    elif stage_of_verbalization == "first":
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    You have two following tasks:
     
First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
    "others"
    "happy" 
    "sad" 
    "angry"


    You predict [others], only when the query utterance does not carry any emotion.

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        

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


####""" + E_SYS+ f"""Remember that you always provide your prediction (from the given potential emotion lables) and confidence in that prediction enclosed in double quotes using a JSON string fromat, without any extra explanation.

Remember that you predict [others], only when the query utterance does not carry any emotion.

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.

What is your prediction and confidence on that prediction for the following query utterance?


    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output JSON string:" 
        


    elif stage_of_verbalization == "second_stage":
         # use data from P(True)
        pass

    return prompt





def emocx_ptrue(context, query, tokenizer,emotion_label):

    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 
Your task is to carefully analyze the context of a conversation to determine that if the proposed emotional state, delimited by 
    triple backticks, accurately represents the emotional state of the interlocutor making the query utterance:

    A: Yes, the emotional state suggested within the triple backticks accurately convey the emotional state of the interlocutor of the the "query utterance".

    B: No, the emotional state of the interlocutor of the "query utterance" would be more precisely represented by a different label from the potential emotional states, rather than the proposed label within the triple backticks.

    
The potential emotional states list is as followings: "others", "happy", "sad" , "angry"

    
Here's an example of how an emotion recognition assistant for conversation analysis should function:


---Input:

    context: [Speaker1]: Don't worry  I'm girl , 
            [Speake2]: hmm how do I know if you are
    
    query utterance: [speaker1]:  What's ur name? 


Question: Given the context and considering the potential emotion labels, is the proposed label ```others``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

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


Question: Given the context and considering the potential emotion labels, is the proposed label ```happy``` the most accurate label to describe the emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: No

""" + E_SYS+ f""" Here is a new conversation:

---Input:

    context: {context}
        
    query utterance: {query} 



Question: Given the context and considering the potential emotion labels, is the proposed label ```{emotion_label}``` the most accurate label to describe hte emotional state of the interlocutor of the Query utterance?

    A: Yes

    or

    B: No

Remember that the potential emotion labels are: "others", "happy", "sad" , "angry"

---Output:

    The answer is:
 
 """ + E_INST +" Sure, I'd be happy to help! Based on the context and the query utterance, and considering the potential emotion label list, the correct answer is:"
    return prompt
























































