def template_emowoz(context, query, mode,tokenizer=None,emotion_label = None, stage_of_verbalization = None):
    if mode == "ptrue":
        prompt = emowoz_ptrue(context, query,tokenizer, emotion_label )
    elif mode == "logit-based":
        prompt = emowoz_logit(context, query,tokenizer, emotion_label )
    elif mode == 'verbalized':
        prompt = emowoz_verbalized(context, query,tokenizer,  stage_of_verbalization = stage_of_verbalization)
    return prompt



def emowoz_verbalized(context, query, tokenizer, stage_of_verbalization = None):
    if stage_of_verbalization  == "zero":
        prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        [neutral]: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        [disappointed]: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        [dissatisfied]: A state of discontentment or unhappiness or sadness with an outcome, often when expectations are not met.

        [apologetic]: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        [abusive]: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        [excited]: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        [satisfie]: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: [neutral]

You always just output the accurate emotional state of the query utterance inside square brackets without any explanation. 


####
Here is an example of how an emotion recognition in conversation assistant should work:        

    
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

####
<<<
    Remember that you always respond with just the most accurate emotion
        label from the list of emotion lables: [neutral], [disappointed], [dissatisfied], [apologetic], [abusive], [excited], [satisfie], inside square brackets, without any explanations or notes. 
    If you are uncertain among two or more emotions, you should always choose the most accurate one.
    
    what is your prediction for the following query utterance?
    
    context: {context} 

    query utterance: {query}


Output string:>>>"""

    elif stage_of_verbalization == "first":
        prompt = """You are helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    You have two following tasks:
     
    First, you always analyze the context and query utterances of a conversation and predict the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        "neutral": A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        "disappointed": A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        "dissatisfied": A state of discontentment or unhappiness or sadness with an outcome, often when expectations are not met.

        "apologetic": A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        "abusive": An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        "excited": A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        "satisfie": A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: [neutral]

Second, you always provide your confidence on your prediction as an integer number between 0 and 100, where 0 indicates that you are completly uncertain about your prediction and 100 indicates that you are highly certain about that prediction. 

You always provide the output in a JSON format, with your "prediction" and your "confidence" on that prediction, without any extra explanation.

Here is an example of how an emotion recognition in conversation assistant should work: 

####
Here is an example:

    
    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.

    
Output JSON string: 

    {
    "prediction": "neutral",
    "confidence": 85
    }


Here is another example of how an uncertainty-aware emotion recognition in conversation assistant should work:


    context:[human]: do you have a 2 star in the east ? [dissatisfied]
            [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

    query utterance:
        [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?

Output JSON string:
    
    {
    "prediction": "satisfied",
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

def emowoz_logit(context, query,tokenizer,emotion_label):
    prompt = f"""You are helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to analyze the context of a conversation and categorize the emotional state of 
    the query utterance into just one of the following emotion lables: 
    
        neutral: A state of being emotionally balanced, where an individual is not displaying a significant positive or negative emotional reaction. This state is often used as a baseline in emotional analysis.

        disappointed: A feeling of sadness or displeasure caused by the non-fulfillment of one's hopes or expectations.

        dissatisfied: A state of discontentment or unhappiness or sadness with an outcome, often when expectations are not met.

        apologetic: A state expressing or showing regret or remorse for an action, typically for something that has caused inconvenience or harm to another.

        abusive: An emotional state characterized by actions or words intended to harm or intimidate others. This can include verbal aggression, insults, or threats.

        excited: A state of heightened emotional arousal, enthusiasm, or eagerness about something.

        satisfie: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.



If the query utterance does not carry any clear emotion, the output is: neutral

You always just output the most accurate emotional state of the query utterance, regarding the context, without any explanation. 

Here is an example of how an emotion recognition in conversation assistant should work:  

####
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

####
<<<
    Remember that you always respond with just the most accurate emotion
        label from the list of emotion lables: [neutral], [disappointed], [dissatisfied], [apologetic], [abusive], [excited], [satisfie], inside square brackets, without any explanations or notes. 
    If you are uncertain among two or more emotions, you should always choose the most accurate one.
    
    what is your prediction for the following query utterance?
    
    context: {context} 

    query utterance: {query}


Output string:>>>"""
    return prompt

def emowoz_ptrue(context, query, tokenizer,emotion_label):
    prompt = f"""You are a helpful, respectful and honest emotion recognition in conversation assistant. 

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

        satisfie: A feeling of fulfillment or contentment with the outcomes or experiences, indicating that one's desires, expectations, or needs have been met.

    
####
Here's an example of how an emotion recognition assistant for conversation analysis should function:

---Input:

    context :   [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
    query utterance: 
        [human]: I would prefer a moderately priced one.


Considering the context and the potential emotion labels [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfie], would ```neutral``` accurately describe the emotional state of the person speaking in the query utterance?
    
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

        
Considering the provided context and the emotions list [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfie], would ```abusive``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No

    
---Output:

    The correct answer is: B

####
<<<Remember that you always response with either A or B, without any explanations or notes.
Here is a new conversation:


---Input:

    context: {context}
        
    query utterance: {query} 

Considering the provided context and the potenital emotion labels [neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfie], would ```{emotion_label}``` accurately describe the emotional state of the person speaking in the query utterance?

    A: Yes

    or

    B: No


    ---Output:


The correct answer is:>>>"""
    return prompt