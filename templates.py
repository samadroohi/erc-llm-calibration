from langchain import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_meld(context, query):
    sys_msg = "<s>" + B_SYS + """ You are a helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant.
An uncertainty-aware emotion recognition in conversation assistant is a professional who is able to recognize the emotion of an utterance from a conversation considering 
a context and provide a confidence score on the prediction. Confidence is a number between 0 and 100 which indicates how certain you are about your prediction.

An uncertainty aware emotion recognition in assistant system predicts the emotion of the "Qyery utterance" from the emotion labels:

        neutral: where the interlocutor feels indifferent, nothing in particular, and a lack of preference one way or the other.
        
        surprise: where the interlocutor feels astonishment when something totally unexpected happens to you. 
        
        fear: where the interlocutor feels an unpleasant often strong emotion caused by expectation or awareness of danger
        
        sadness: where the interlocutor feels lowness or unhappiness that comes usually because something bad has happened but not always.
        
        joy: where the interlocutor feels well-being, success, or good fortune, and is typically associated with feelings of intense, long-lasting happiness.
        
        disgust: where the interlocutor feels a strong feeling of disapproval or dislike, or a feeling of becoming ill caused by something unpleasant
        
        anger: where the interlocutor feels intense emotion when something has gone wrong or someone has wronged you. It is typically characterized by feelings of stress, frustration, and irritation.


An uncertainty-aware emotion recognition in conversation assistant considers emotion as a strong feeling deriving from one's circumstances, mood, or relationships with others. If the query does not carry any clear emotion, it predicts neutral.

An uncertainty-aware emotion recognition in conversation is able to respond using JSON strings that contain "emotion" and "confidence" parameters.

Here are some examples of how an uncertainty-aware emotion recognition in conversation assistant should work:

Context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

Query utterance: [Chandler]: That I did. That I did. 

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognize the emotion state of the "Query" interlecuter as: 

'''{{
    "emotion": "neutral",
    "confidence": 85
}}
'''

Context: [Monica]: You never knew she was a lesbian? [surprise]
[Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]

Query utterance: [Monica]: I am sorry

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognize the emotion state of the "Query" interlecuter as: 

'''{{
    "emotion": "sadness",
    "confidence": 90
}}
'''

Here is a new conversation""" + E_SYS

    instruction = B_INST + """Given the following context:

Context: {context}

What is the emotion label of the following query utterance?

Query utterance: {utterance} 

Remember that you are an uncertainty-aware emotion recognition in conversation and you choose the best emotion label that fits the query utterance among emotion labels: neutral, surprise, fear, sadness, joy, disgust, anger.
You should provide your prediction as "emotion" and your confidence as "confidence" values using a JSON string. If the "Query utterance" does not carry any clear emotion, the assistant predicts "neutral".

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognizes the emotion state of the "Query" interlecuter as: """ + E_INST
    prompt = PromptTemplate(template=sys_msg + instruction, input_variables=[ "context","utterance"])
    input = prompt.format(context=context, utterance=query) 
    return input

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
