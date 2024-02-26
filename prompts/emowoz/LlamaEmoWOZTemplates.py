B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_emowoz(context, query, mode,tokenizer=None,emotion_label = None, stage_of_verbalization = None):
    if mode == "P(True)":
        prompt = emowoz_ptrue(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = emowoz_verbalized(context, query,tokenizer,  stage_of_verbalization = stage_of_verbalization)
    return prompt



def emowoz_verbalized(context, query, tokenizer, stage_of_verbalization = None):
    if stage_of_verbalization  == "zero":

        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation between a human and a customer support service agent and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        [neutral]: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        [disappointed]: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        [dissatisfied]: A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        [apologetic]: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        [abusive]: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        [excited]: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        [satisfie]: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.


If the query utterance does not carry any emotion, the output is: [neutral]

If you are uncertain among two or more emotions, you should always choose the most accurate one.

You always will respond with the most accurate emotional state of the query utterance. 

Your always respond with just the most accurate emotion lable (single lable) without any explanations or notes on the output. 


Here is an example of how an emotion recognition in conversation assistant should work:        

####
Here is an examples:
    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.

    
Output string: [neutral]


Here is another example of how an emotion recognition in conversation assistant should work:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?


Output string: [satisfied]

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion label (single lable) without any explanations or notes. If you are uncertain among two or more emotions, you should always choose the most accurate one.
 
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
        
    elif stage_of_verbalization == "first":
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
You have two following tasks:
     
First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        "neutral": A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        "disappointed": A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        "dissatisfied": A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        "apologetic": A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        "abusive": An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        "excited": A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        "satisfie": A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: [neutral]

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        

####
Here is an examples:
    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.

    
Output JSON string: 
    
    {
    "prediction": "neutral",
    "confidence": 85
    }


Here is another example of how an emotion recognition in conversation assistant should work:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?

Output JSON string:
    
    {
    "prediction": "satisfied",
    "confidence": 90
    }


####""" + E_SYS+ f"""Remember that you always provide your prediction (from the given potential emotion lables) and confidence in that prediction enclosed in double quotes using a JSON string fromat, without any extra explanation.

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.


    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output JSON string:" 
        


    elif stage_of_verbalization == "second_stage":
         # use data from P(True)
        pass

    return prompt





def emowoz_ptrue(context, query, tokenizer,emotion_label):

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
























































