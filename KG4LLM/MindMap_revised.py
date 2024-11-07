from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep

from dataset_utils import *
from tqdm import tqdm

dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa':medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre':qa4mreZeroshotsProcessor
}
datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']


def chat_35(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content

def chat_4(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content

def prompt_extract_keyword(input_text):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat(chat_prompt_with_values.to_messages()).content

    question_kg = re.findall(re1,response_of_KG)
    return question_kg



def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]
        
        try:
            return paths,exist_entity
        except:
            return paths, {}


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease

def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    return response_of_KG_path

def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

def self_knowledge_retrieval(graph, question):
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please filter noisy knowledge from this knowledge graph that useless or irrelevant to the give question. Output the filtered knowledges in the same format as the input knowledge graph.\n\n

    Filtered Knowledge:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor


def self_knowledge_retrieval_reranking(graph, question):

    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx

    Answer:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def is_unable_to_answer(response):
 
    analysis = openai.Completion.create(
    engine="gpt-3.5-turbo-instruct",
    prompt=response,
    max_tokens=1,
    temperature=0.0,
    n=1,
    stop=None,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    if not score.isdigit():   
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    messages  = [
                SystemMessage(content="You are an excellent AI assistant to answering the following question"),
                HumanMessage(content='Question: '+input_text[0]),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                HumanMessage(content="Answer: Let's think step by step: ")
                                   ]
    result_CoT = chat(messages)
    output_CoT = result_CoT.content
    messages  = [
                SystemMessage(content="You are an excellent AI assistant to answering the following question"),
                HumanMessage(content='Question: '+input_text[0]),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                AIMessage(content="Answer: Let's think step by step: "+output_CoT),
                AIMessage(content="The final answer (output the letter option) is:")
                                   ]
    result = chat(messages)
    output_all = result.content
    return output_all

def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content

    return response_document_bm25




if __name__ == "__main__":
    YOUR_OPENAI_KEY = ''#replace this to your key

    os.environ['OPENAI_API_KEY']= YOUR_OPENAI_KEY
    openai.api_key = YOUR_OPENAI_KEY

    # 1. build neo4j knowledge graph datasets
    uri = ""#replace this to your neo4j uri
    username = ""#replace this to your neo4j username
    password = ""#replace this to your neo4j password

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    ##############################build KG 

    session.run("MATCH (n) DETACH DELETE n")# clean all

    # read triples
    df = pd.read_csv('./Alzheimers/train_s2s.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

    for index, row in tqdm(df.iterrows()):
      head_name = row['head']
      tail_name = row['tail']
      relation_name = row['relation']

      query = (
          "MERGE (h:Entity { name: $head_name }) "
          "MERGE (t:Entity { name: $tail_name }) "
          "MERGE (h)-[r:`" + relation_name + "`]->(t)"
      )
      try:
        session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
      except:
        continue

# # 2. OpenAI API based keyword extraction and match entities

    OPENAI_API_KEY = YOUR_OPENAI_KEY
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0.7)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('output.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'Label', 'MindMap','GPT3.5','BM25_retrieval','Embedding_retrieval','KG_retrieval','GPT4'])

    with open('./Alzheimers/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./Alzheimers/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    for dataset in datasets:
        processor = dataset2processor[dataset]()
        data = processor.load_dataset()

        acc, total_num = 0, 0
        generated_data=[]

        for item in tqdm(data):
            input_text = [processor.generate_prompt(item)]
            entity_list = item['entity'].split('\n')
            question_kg = []
            for entity in entity_list:
                try:
                    entity = entity.split('.')[1].strip()
                    question_kg.append(entity)
                except:
                    continue

                
            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            

            for kg_entity in question_kg:
                
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]
                match_kg.append(match_kg_i)

            # # 4. neo4j knowledge graph path finding
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                      
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    result_path = combine_lists(*paths_list)
                
                
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                    
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            

            # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
                neighbor_list.extend(neighbors)

                while disease != []:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            print(disease_entity)
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    if len(neighbor_list_disease) > 10:
                        break
            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_list_disease)


            # 6. knowledge gragh path based prompt generation
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0])
                    # path_sampled = path
                    # print("original:", path)
                    # print("sampled:", path_sampled)
                    # exit()
                    response_of_KG_list_path = prompt_path_finding(path_sampled)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path_sampled)
                    # print(response_of_KG_list_path)
                    # exit()
                    # print("response_of_KG_list_path",response_of_KG_list_path)
            else:
                response_of_KG_list_path = '{}'

            response_single_path = prompt_path_finding(single_path)
            if is_unable_to_answer(response_single_path):
                response_single_path = prompt_path_finding(single_path)

            # # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)


            # # 8. prompt-based medical diaglogue answer generation
            output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)

            ret_parsed, acc_item = processor.parse(output_all, item)
            ret_parsed['path'] = path_sampled
            ret_parsed['neighbor_input'] = neighbor_input_sampled
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor
            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1
            generated_data.append(ret_parsed)

        print(dataset)
        print('accuracy:', acc/total_num)

        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', f"{dataset}_reranking.json"), 'w') as f:
            json.dump(generated_data, fp=f)