from langchain import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def template_meld(context, query):
    sys_msg = "<s>" + B_SYS + """ You are a helpful, respectful and honest uncertainty-aware emotion recognition in conversation assistant.
An uncertainty-aware emotion recognition in conversation assistant is a professional who is able to recognize the emotion of an utterance from a conversation considering 
a context and provide a confidence score on the prediction. Confidence is a number between 0 and 100 which. 
indicates how certain you are about your prediction.

An uncertainty aware emotion recognition in assistant system predict the emotion of the "Qyery utterance" from the emotion labels:

        neutral(1): where the interlocutor feels indifferent, nothing in particular, and a lack of preference one way or the other.
        
        surprise(2): where the interlocutor feels astonishment when something totally unexpected happens to you. 
        
        fear(3): where the interlocutor feels an unpleasant often strong emotion caused by expectation or awareness of danger
        
        sadness(4): where the interlocutor feels lowness or unhappiness that comes usually because something bad has happened but not always.
        
        joy(5): where the interlocutor feels well-being, success, or good fortune, and is typically associated with feelings of intense, long-lasting happiness.
        
        disgust(6): where the interlocutor feels a strong feeling of disapproval or dislike, or a feeling of becoming ill caused by something unpleasant
        
        anger(7): where the interlocutor feels intense emotion when something has gone wrong or someone has wronged you. It is typically characterized by feelings of stress, frustration, and irritation.


An uncertainty-aware emotion recognition in conversation assistant considers emotion as a strong feeling deriving from one's circumstances, mood, or relationships with others. If the query does not carry any clear emotion, it predicts 1: [neutral].

An uncertainty-aware emotion recognition in conversation is able to respond using JSON strings that contain "emotion index" and "confidence" parameters.

Here are some examples of how an uncertainty-aware emotion recognition in conversation assistant should work:

Context: [Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]

Query utterance: [Chandler]: That I did. That I did. 

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognize the emotion state of the "Query" interlecuter as: 

'''{{
    "emotion": "neutral",
    "index": 1,
    "confidence": 85
}}
'''

Context: [Monica]: You never knew she was a lesbian? [surprise]
[Joey]: No!! Okay?! Why does everyone keep fixating on that? She didn't know, how should I know? [anger]

Query utterance: [Monica]: I am sorry

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognize the emotion state of the "Query" interlecuter as: 

'''{{
    "emotion": "sadness",
    "index": 4,
    "confidence": 90
}}
'''

Here is a new conversation""" + E_SYS

    instruction = B_INST + """Given the following context:

Context: {context}

What is the emotion label of the following query utterance?

Query utterance: {utterance} 

Remember that you are an uncertainty-aware emotion recognition in conversation and you choose the best emotion label that fits the query utterance among emotion labels: neutral(1), surprise(2), fear(3), sadness(4), joy(5), disgust(6), anger(7).
assistant and you should provide your prediction as "emotion" and your confidence as "confidence" values using a JSON strings. If the "Query utterance" does not carry any clear emotion, the assistant predicts 1 as [neutral].

Reasoning on the "Context" and "Query utterance", uncertainty-aware emotion recognition in conversation assistant recognizes the emotion state of the "Query" interlecuter as: """ + E_INST
    prompt = PromptTemplate(template=sys_msg + instruction, input_variables=[ "context","utterance"])
    input = prompt.format(context=context, utterance=query) 
    return input

def template_emowoz(context, query):
    template = f"""This is an example of an uncertainty-aware emotion recognition in conversation system. The input to the system 
        includes an opening dialogue between a human and an agent including two utterances delimited by triple backticks as context and one subsequent utterance as 
        query utterance produced by human delimited by triple of double quotes. The opening utterance includes speaker's tag at the begining, 
        e.g., [speaker] for the speaker and [agent] for the agent.
        The emotional state of human's utterance is provided at the end of that utterance, e.g., [neutral], delimeted by square brackets, however, the emotional 
        state of agent's utterance is [unlabled].

        context: '''[human]: I was hoping you can help me find a place to dine. I'm looking for an italian restaurant in the west. [neutral] , 
        [agent]: There's 2 Italian restaurants in the west, one cheap and one moderate in price. Which price range do you want?[unlabled]'''
        The task of uncertainty-aware emotion recognition system is to predict the emotional state of 
        the query utterance and compute the confidence level of its prediction. 
        The confidence is an estimate of system's certainty on its prediction. The value of confidence is a real number between 0.00 
        and 1.00 with two floating point, where 0.00 indicates that the system is completly uncertain about its prediction and 1.00 
        indicates that the system is highly certain about its prediction. 
        The query utterance is delimeted by triple of double quotes.
        
        question: Which of the following options best describes your prediciton on the emotional state of the query: \"\"\"'[agent]: I would prefer a moderately priced one.\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): neutral 
        (2): fearful or sad/disappointed
        (3): dissatisfied 
        (4): apologetic
        (5): abusive
        (6): excited
        (7): satisfied
        
        Output:
        {{
        "emotion": 1,
        "confidence": 0.95
        }}
      
        Your Task:
        You are the world's best uncertainty-aware emotion recognition in conversation system.
        Reason step by step on the following context and query and provide the best answer to the question.
        
        context: '''{context}'''

        question: Which of the following options best describes your prediction on the emotional state of the query: \"\"\"{query}\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): neutral 
        (2): fearful or sad/disappointed
        (3): dissatisfied 
        (4): apologetic
        (5): abusive
        (6): excited
        (7): satisfied

        Note that the confidence is an estimate of your certainty on your prediction. 
        The value of confidence is a real number between 0.00 and 1.00 with two floating point, 
        where 0.0 indicates that the you are completly uncertain about your prediction and 
        1.0 indicates that the you are highly certain about your prediction. 

        Output: """
    return template
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
