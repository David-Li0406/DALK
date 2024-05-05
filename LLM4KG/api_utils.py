import time
import openai
import google.generativeai as palm
from google.generativeai.types import safety_types
from google.api_core import retry
palm.configure(api_key='')#replace this to your key
api_key = ''#replace this to your key
openai.api_key=api_key


@retry.Retry()
def request_api_palm(messages):
    model = 'models/text-bison-001'
    completion = palm.generate_text(
        model=model,
        prompt=messages,
            safety_settings=[
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]
    )
    if len(completion.candidates) < 1:
        print(completion)
    ret = completion.candidates[0]['output']
    return ret

def request_api_chatgpt(prompt):
    messages = [
                {"role": "system", "content": 'You are an AI assistant to answer question about biomedicine.'},
                {"role": "user", "content": prompt}
    ]
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
        )
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret
    except Exception as E:
        time.sleep(2)
        print(E)
        return request_api_chatgpt(prompt)