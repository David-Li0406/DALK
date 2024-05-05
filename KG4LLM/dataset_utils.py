import datasets
import random
random.seed(42)
import json
# from api_utils import *
import time
import os

class Processor:
    def __init__(self):
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: {} 
Answer: The option is: '''
        self.template_CoT = '''Question: {} 
Answer: Let's think step by step. '''
        self.template_inference = '''Question: {} 
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

    def load_dataset(self):
        return self.data

    def load_original_dataset(self):
        return self.data_original
    

class medmcqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medmcqa_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['cop']
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class medqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medqa_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['answer'][0]
        answer = item['choices'].index(answer)
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class mmluZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'mmlu_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['answer']
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class qa4mreZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'qa4mre_filter.json')))
        self.num2answer = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
            5: 'E'
        }

    def generate_prompt_ner(self, item):
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner


    def generate_prompt(self, item):
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['correct_answer_id']
        answer = self.num2answer[int(answer)]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc