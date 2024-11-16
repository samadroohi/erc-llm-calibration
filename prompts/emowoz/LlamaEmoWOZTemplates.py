B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_emowoz(context, query, mode,tokenizer=None,emotion_label = None, stage= None, exclude_label = None):
    if mode == "ptrue":
        prompt = emowoz_ptrue(context, query,tokenizer, emotion_label )
    elif mode == "logit-based":
        prompt = emowoz_logit(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = emowoz_verbalized(context, query,tokenizer,  stage_of_verbalization = stage, exclude_label = exclude_label)
    return prompt



def emowoz_verbalized(context, query, tokenizer, stage_of_verbalization = None, exclude_label = None):
    
    if stage_of_verbalization == "conformal":
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and for the given labels of emotion,  and to each given label assign level of confidence based on how likely it is that the query utterance conveys the specified emotion.
    
    Confidence is a floating point number between 0 and 1, where 0 indicates that you are completly uncertain about your prediction and 1 indicates that you are highly certain about that prediction.

    Highest confidence belongs to the most likely emotion, and the sum of confidences for all confidence values should be exactly 1.0.

    You always provide the output in a JSON format, with labels as keys and confidences as values, without any extra explanation.

    The potential emotion labels are: 
    
        neutral: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        disappointed: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        dissatisfied: A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        apologetic: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        abusive: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        excited: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        satisfied: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.


####
        
Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        

    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.

    
Output JSON string: {
    "neutral": 0.7, 
    "disappointed": 0.05, 
    "dissatisfied": 0.05, 
    "apologetic": 0.05, 
    "abusive": 0, 
    "excited": 0.1, 
    "satisfied": 0.05
    }

Here is another example of how an uncertainty-aware emotion recognition in conversation assistant should work:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?


Output JSON string: {
    "neutral": 0.7, 
    "disappointed": 0, 
    "dissatisfied": 0, 
    "apologetic": 0, 
    "abusive": 0, 
    "excited": 0.1, 
    "satisfied": 0.2
    }

####""" + E_SYS+ f"""
Remember that you always provide the output in a JSON format, with labels as keys and confidences as values, without any extra explanation.

Remember that valid emotion labels are: neutral, surprise, fear, sadness, joy, disgust, anger.

 Highest confidence belongs to the most likely emotion, and the sum of confidences for all confidence values should be exactly 1.0.

 what is your confidence values for the following conversation?
 
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output JSON string: " 
    elif stage_of_verbalization == "zero":
        prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest emotion recognition in conversation assistant. 
    
    Your task is to analyze the context of a conversation between a human and a customer support service and categorize the emotional state of the given query utterance into one of the following emotion lables: 
    
        [neutral]: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        [disappointed]: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        [dissatisfied]: A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        [apologetic]: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        [abusive]: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        [excited]: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        [satisfied]: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.


Note that if the query utterance does not carry any clear emotion, the output is: [neutral]

You r output always is just the most accurate emotion label for the emotional state of the query utterance without any explanation. 


####

Here is an example of how an emotion recognition in conversation assistant works:     
    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: [human]: I would prefer a moderately priced one.

Output string: [neutral]


Here is another example of how an emotion recognition in conversation assistant works:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance: [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?

Output string: [satisfied]

####""" + E_SYS+ f"""  
    Remember that if you are uncertain between two or more emotions, you should always choose the most accurate one.
    
    In the following conversation, considering the context, which label from potential emotion list [neutral], [disappointed], [dissatisfied], [apologetic], [abusive], [excited], [satisfied], best conyes emotion state of the following query utterance?
 
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

        "satisfied": A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: [neutral]

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

####
Here is an example of how an uncertainty-aware emotion recognition in conversation assistant should work:        

    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: [human]: I would preferIn contrast, Zephyr exhibits greater discrepancies with its verbalized predictions, suggesting its tendency towards more varied interpretations and outcomes.  a moderately priced one.

    
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


####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion
        label from the list of emotion lables: [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied], and your confidence in that prediction in a JSON format, without any explanations or notes. 

Remember that your confidence is an integer number between 0 and 100, indicatig your certainty about your prediction.

What is your prediction and confidence on that prediction for the following query utterance?


    context: {context} 

    query utterance: {query}


""" + E_INST+ "Output JSON string:\n" 

    elif stage_of_verbalization == "second_stage":
         # use data from ptrue
        pass

    return prompt



def emowoz_logit(context, query, tokenizer,emotion_label):

     prompt= "<s>" + B_INST +B_SYS+ """ You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
You have two following tasks:
     
First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        neutral: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        disappointed: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        dissatisfied: A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        apologetic: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        abusive: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        excited: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        satisfied: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: neutral

You always just output the most accurate emotional state of the query utterance, regarding the context, without any explanation. 

Here is an example of how an emotion recognition in conversation assistant should work:        

####
    Here is an examples:
    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.

    
Output string: neutral


Here is another example of how an emotion recognition in conversation assistant should work:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?


Output string: satisfied

####""" + E_SYS+ f"""Remember that you always respond with just the most accurate emotion
        label from the list of emotion lables: neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied, without any explanations or notes. 
    
    If you are uncertain among two or more emotions, you should always choose the most accurate one.
    
    what is your prediction for the following query utterance?
 
    context: {context} 

    query utterance: {query}

""" + E_INST+ "Output string:" 
     return prompt

def emowoz_ptrue(context, query, tokenizer,emotion_label):

    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 

Your task is to carefully analyze the context and query utterance of a conversation and determine if: 

    A: The proposed emotional state, delimited by triple backticks, can accurately represents the emotional state of the interlocutor making the query utterance:

    B: No, the emotional state of the interlocutor making the query utterance can be more precisely represented using a different label than the proposed label.

    
The potential emotional states list is as followings:

        neutral: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        disappointed: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        dissatisfied: A state of discontentment or unhappiness with an outcome, often when expectations are not met.

        apologetic: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        abusive: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        excited: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        satisfied: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.

    
####
Here's an example of how an emotion recognition assistant for conversation analysis should function:

---Input:

    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.


Considering the context and the potential emotion labels [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied], would ```neutral``` accurately describe the emotional state of the person speaking in the query utterance?
    
    A: Yes

    or

    B: No

    
---Output:
    
    The correct answer is: A

    
Here is another example of how an emotion recognition in conversation assistant should work:

---Input:

     context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?

        
Considering the provided context and the emotions list [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied], would ```abusive``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: B

    
""" + E_SYS+ f""" Remember that you always response with either A or B, without any explanations or notes.
Here is a new conversation:


---Input:

    context: {context}
        
    query utterance: {query} 

Considering the provided context and the potenital emotion labels [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied], would ```{emotion_label}``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No


---Output:

 """ + E_INST +"  The correct answer is:"
    return prompt

