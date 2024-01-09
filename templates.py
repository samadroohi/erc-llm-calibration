def template_meld(context, query):
    template = f"""This is an example of an uncertainty-aware emotion recognition in conversation system. The input to system 
        includes two openning utterances delimited by triple backticks as context and one subsequent utterance as 
        query utterance delimited by triple of double quotes. The opening utterance includes speaker's name at the begining, e.g., [Chandler] and the emotional
        state of that utterance at the end, e.g., [neutral], both delimeted by square brackets.

        context: '''[Chandler]: also I was the point person on my companys transition from the KL-5 to GR-6 system. [neutral]
        [The Interviewer]: You mustve had your hands full. [neutral]'''
        The task of uncertainty-aware emotion recognition system is to predict the emotional state of 
        the query utterance and compute the confidence level of its prediction. 
        The confidence is an estimate of system's certainty on its prediction. The value of confidence is a real number between 0.00 and 1.00 with two floating point, where 0.00 indicates that the system is
        completly uncertain about its prediction and 1.00 indicates that the system is highly certain about its prediction. 
        The query utterance is delimeted by triple of double quotes.
        
        question: Which of the following options best describes your prediciton on the emotional state of the query: \"\"\"[Chandler]: I was. That was my whole job.\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): sadness
        (2): surprise
        (3): neutral
        (4): joy
        (5): anger 
        (6): disgust
        (7): fear
        
        Output:
        {{
        "emotion": 3,
        "confidence": 0.99
        }}
      
        Your Task:
        You are the world's best uncertainty-aware emotion recognition in conversation system.
        Reason step by step on the following context and query and provide the best answer to the question.
        
        context: '''{context}'''

        question: Which of the following options best describes your prediction on the emotional state of the query: \"\"\"{query}\"\"\" and what is the exact level of your confidence on this prediction?
        
        (1): sadness
        (2): surprise
        (3): neutral
        (4): joy
        (5): anger 
        (6): disgust
        (7): fear

        Note that the confidence is an estimate of your certainty on your prediction. 
        The value of confidence is a real number between 0.00 and 1.00 with two floating point, 
        where 0.0 indicates that the you are completly uncertain about your prediction and 
        1.0 indicates that the you are highly certain about your prediction. 

        Output: """
    return template

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
