from dataset_utils import *

import json
import os
from tqdm import tqdm
import time
import argparse
import openai
api_key = ''#replace this to your key
openai.api_key=api_key

dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa': medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre':qa4mreZeroshotsProcessor
}

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

def main():
    for dataset in ['medqa', 'medmcqa', 'mmlu', 'qa4mre']:
        processor = dataset2processor[dataset]()
        data = processor.load_original_dataset()
        generated_data = []
        acc, total_num = 0, 0
        for item in tqdm(data):
            time.sleep(2)
            prompt = processor.generate_prompt_ner(item)
            ret = request_api_chatgpt(prompt)
            item['entity'] = ret
            generated_data.append(item)
        with open(os.path.join('Alzheimers', 'result_ner', f"{dataset}_zero-shot.json"), 'w') as f:
            json.dump(generated_data, fp=f)

if __name__ == '__main__':
    main()