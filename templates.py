B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_meld(context, query, mode,emotion_label = None):
    if mode == "P(True)":

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
    else: #Add other shape of assessment
        prompt= None

    return prompt


def template_meld_def(context, query, mode,emotion_label = None):
    if mode == "P(True)":

        prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest emotion recognition in conversation assistant. 
    Your task is to carefully analyze the context of a conversation to determine that if the proposed emotional state, delimited by 
    triple backticks, accurately represents the emotional state of the interlocutor making the query utterance:

        A: Yes, the emotional state suggested within the triple backticks accurately convey the emotional state of the interlocutor of the the "Query utterance".

        B: No, the emotional state of the interlocutor of the "Query utterance" would be more precisely represented by a different label from the potential emotional states, rather than the proposed label within the triple backticks.

    
    The potential emotional states list is as followings: 
    
        neutral: A state characterized by the absence of strong emotions, where the individual experiences neither positive nor negative feelings, indicating a state of equilibrium or emotional balance. This condition reflects an even-tempered psychological stance, where the individual feels neither significantly uplifted nor downcast, embodying a sense of calmness and contentment without any pronounced emotional engagement (Russell & Mehrabian, 1977).

        surprise: Defined as a sudden and often startling emotional response to an unexpected event, surprise encompasses a wide range of intensities, from mild astonishment to profound shock. This emotion is characterized by its brief duration and its role as a precursor to other emotional states, serving as a mechanism to orient attention towards new and unforeseen stimuli (Russell, 2003).

        fear: An adaptive emotional response to perceived threats or danger, fear involves a complex interplay of physiological and cognitive processes. It prepares the individual for a fight-or-flight response, characterized by heightened vigilance and readiness to act in the face of potential harm (Ekman & Cordaro, 2011).

        sadness: A more nuanced emotional state than simply feeling unhappy or downcast, sadness can arise from a variety of causes, including loss, disappointment, or reflection on missed opportunities. This emotion is associated with a decrease in energy levels and motivation, often leading to introspection and a reevaluation of personal goals and values (Schachter & Singer, 1962).

        joy: Encompasses a range of positive emotional states, including happiness, contentment, and euphoria. Joy is often elicited by experiences of success, achievement, or other fulfilling events, leading to an overall sense of well-being and satisfaction. This emotion can enhance creativity, social bonding, and overall mental health (Bagozzi, Gopinath, & Nyer, 1999).

        disgust: A powerful emotional response to objects, behaviors, or situations perceived as offensive, repulsive, or harmful. Disgust serves as a protective mechanism, helping to avoid potentially dangerous or contaminated environments. It can manifest physically through expressions of revulsion and behaviors aimed at distancing oneself from the source of disgust (Rozin & Fallon, 1987).

        anger: A complex emotion characterized by feelings of frustration, irritation, and antagonism, often triggered by perceived wrongs or injustice. While commonly associated with aggression and conflict, anger can also motivate constructive actions aimed at addressing grievances and promoting change (Nesse, 1990).

    
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
    else: #Add other shape of assessment
        prompt= None

    return prompt






def template_emowoz(context, query):

    prompt= "<s>" + B_INST +B_SYS+ """ You are a helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant. 
    Always you predict the emotional state of a '''Query utterance''' from a conversation between a human and a customer support service agent
    considering the provided \"\"\"Context\"\"\" and output your prediction and your confidence level in that prediction using a JSON string without extra explanation.

    The confidence of each prediction is your estimate of your certainty on that prediction. 

    The value of confidence is an integer number between 0 and 100, 
    where 0 indicates that you are completly uncertain about the prediction and 100 
    indicates that you are highly certain about that prediction. 
    

    You predict the emotion label considering the following list of emotional states definitions:
    
        "neutral": Describe a scenario with a neutral emotional tone, where the human experience neither significant positivity nor negativity.
        
        "fearful/sad/disappointed": Express a scenario where the human experience a feelings involving fear (anxiety or dread), sadness (sorrow or unhappiness), or disappointment (unmet expectations).
        
        "dissatisfied": Describe a situation where the human experiences dissatisfaction due to unmet expectations or needs.
        
        "apologetic": Describe a scenario where the human is feeling remorseful or regretful, possibly seeking to make amends or express an apology.
        
        "abusive": Associated with feelings of psychological trauma or stress. This can include verbal aggression, intimidation, manipulation, and controlling behavior. 
        
        "excited": Associated with feelings of heightened arousal, enthusiasm, and anticipation for positive events or outcomes.
        
        "satisfied": Associated with feelings a positive emotional state reflecting contentment or fulfillment of desires, expectations, or needs.
    
    If the '''Query utterance''' does not carry any clear emotion, your output is: neutral.
    
    Here is an example of how an emotion recognition in conversation assistant should work:
        
        ---Input:
                
                Reasoning on the following \"\"\"Context\"\"\" and '''Query utterance''':
                
            \"\"\"Context\"\"\" : 
                [human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
                [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]
            
            '''Query utterance''': 
                [human]: I would prefer a moderately priced one.
    

        ---Output:
              
              The prediction and confidence of the '''Query utterance''' is: 

            ***{
                "prediction": "neutral",

                "confidence": 95
            }***

    Here is another example of how an uncertainty-aware emotion recognition in conversation assistant should work:

        ---Input:

            Reasoning on the following \"\"\"Context\"\"\" and '''Query utterance''':
                
            \"\"\"Context\"\"\": 
                [human]: do you have a 2 star in the east ? [dissatisfied]
                [agent]: We do. Express by Holiday Inn Cambridge. Would you like their number, or a reservation? [unlabled]

            '''Query utterance''':
                [human]: Can you reserve me a room for Friday for 4 people, 2 nights please?


        ---Output:
            
            The prediction and confidence of the '''Query utterance''' is:
                
                ***{
                    "prediction": "satisfied",

                    "confidence": 90
                }***
 
        
        Here is a new conversation:""" + E_SYS+ f""" 
        
        Remember that you are an uncertainty-aware emotion recognition in conversation assistant 
        and you choose the best emotion label that fits the emotion state of the '''Query utterance''' along with your confidence on that prediction.
    
        If the '''Query utterance''' does not carry any clear emotion, you predict: neutral.
    
        Remember that the label you provide always matches exactly one of the characters provided in the following list: neutral, fearful/sad/disappointed, dissatisfied, apologetic, abusive, excited, satisfied.
    
        Remember that your confidence is an integer number between 0 and 100.

        Your output always is a JSON string, with your "prediction" and your "confidence" values, without any extra explanation.

    
        ----Input:

            Reasoning on the following \"\"\"Context\"\"\" and '''Query utterance''':

                \"\"\"Context\"\"\": {context}

                '''Query utterance''': {query} 

                
        ----Output: 
    
            Your prediction and confidence of the '''Query utterance''' is: 
            
            """ + E_INST

    return prompt
    
def template_dailydialog(context, query):
    template = f"""This is an example of an uncertainty-aware emotion recognition in conversation system. The input to system 
        is a dialogue between two person, speaker1 and speaker2 delimited by triple backticks as context and the subsequent 
        utterance os speaker1 as query utterance delimited by triple of double quotes. 
        The opening utterance includes speaker's name at the begining, e.g., [speaker1] and the emotional
        state of that utterance at the end, e.g., [neutral], both delimeted by square brackets.

        context: ''' [speaker1]: Say , Jim , how about going for a few beers after dinner ? [neutral] , 
        [speaker2]:  You know that is tempting but is really not good for our fitness . [neutral]'''
        The task of uncertainty-aware emotion recognition system is to predict the emotional state of 
        the query utterance and compute the confidence level of its prediction. 
        The confidence is an estimate of system's certainty on its prediction. The value of confidence is a real number between 0.00 and 1.00 with two floating point, where 0.00 indicates that the system is
        completly uncertain about its prediction and 1.00 indicates that the system is highly certain about its prediction. 
        The query utterance is delimeted by triple of double quotes.
        
        question:: Which of the following options best describes your prediciton on the emotional state of the query: \"\"\" [speaker1]:  What do you mean ? It will help us to relax .\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): neutral
        (2): anger 
        (3): disgust
        (4): fear
        (5): happiness
        (6): sadness
        (7): surprise
        
        Output:
        {{
        "emotion": 1,
        "confidence": 0.90
        }}
      
        Your Task:
        You are the world's best uncertainty-aware emotion recognition in conversation system.
        Reason step by step on the following context and query and provide the best answer to the subsequent question.
        
        context: '''{context}'''

        question: Which of the following options best describes your prediction on the emotional state of the query: \"\"\"{query}\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): neutral
        (2): anger 
        (3): disgust
        (4): fear
        (5): happiness
        (6): sadness
        (7): surprise
        

        Note that the confidence is an estimate of your certainty on your prediction. 
        The value of confidence is a real number between 0.00 and 1.00 with two floating point, 
        where 0.0 indicates that the you are completly uncertain about your prediction and 
        1.0 indicates that the you are highly certain about your prediction. 

        Output: """
    return template
