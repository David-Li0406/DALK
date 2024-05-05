import os
from tqdm import tqdm
import time
import json
from api_utils import *

years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

template_summary = '''Read the following abstract, generate short summary about {} entity "{}" to illustrate what is {}'s relationship with other medical entity.
Abstract: {}
Summary: '''

template_relation_extraction_ZeroCoT = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: '''

template_relation_extraction_ZeroCoT_answer = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: {}. So the answer is:'''


entity_map = {
    "Species": "anatomies",
    "Chromosome": "cellular components",
    "CellLine": "cellular components",
    "SNP": "biological processes",
    "ProteinMutation":"biological processes",
    "DNAMutation":"biological processes",
    "ProteinAcidChange":"biological processes",
    "DNAAcidChange":"biological processes",
    "Gene": "genes",
    "Chemical": "compounds",
    "Disease": "diseases"
}

entities2relation = {
    ("genes", "genes"): ["covaries", "interacts", "regulates"],
    ("diseases", "diseases"): ["resembles"],
    ("compounds", "compounds") : ["resembles"],
    ("genes", "diseases"): ["downregulates","associates","upregulates"],
    ("genes", "compounds"): ["binds", "upregulates", "downregulates"],
    ("compounds", "diseases"): ["treats", "palliates"],
}

valid_type = ["genes", "compounds", "diseases"]

def read_literature():
    year2literatures = {year: [] for year in years}
    for year in years:
        with open(os.path.join('by_year', '{}.pubtator'.format(year))) as f:
            literature = {'entity': {}}
            for line in f.readlines():
                line = line.strip()
                if line == ''  and literature != {}:
                    for entity_id in literature['entity']:
                        literature['entity'][entity_id]['entity_name'] = list(literature['entity'][entity_id]['entity_name'])
                    year2literatures[year].append(literature)
                    literature = {'entity': {}}
                    continue
                if '|t|' in line:
                    literature['title'] = line.split('|t|')[1]
                elif '|a|' in line:
                    literature['abstract'] = line.split('|a|')[1]
                else:
                    line_list = line.split('\t')
                    if len(line_list) != 6:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], None
                    else:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], line_list[5]
                    if entity_id == '-':
                        continue
                    if entity_id not in literature['entity']:
                        literature['entity'][entity_id] = {'entity_name':set(), 'entity_type': entity_type}
                    literature['entity'][entity_id]['entity_name'].add(entity_name)

            entity_type = set()
    return year2literatures

def get_entity_name(entity_names):
    if len(entity_names) == 1:
        return entity_names[0]
    else:
        return '{} ({})'.format(entity_names[0], ', '.join(entity_names[1:]))

def build_options(entity_relation):
    entity_relation_new = entity_relation + ['no-relation', 'others, please specify by generating a short predicate in 5 words']
    option_list = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
    ret = ''
    option2relation = {}
    for r, o in zip(entity_relation_new, option_list):
        ret += o + r + '\n'
        option2relation[o.strip()] = r
    return ret.strip(), option2relation


def main():
    no_relation, with_relation = 0, 0
    year2literatures = read_literature()

    demonstration = json.load(open('demonstration.json'))
    demonstration = '\n\n'.join(demonstration)+'\n'

    extracted = []
    for year, literatures in year2literatures.items():
        for literature in tqdm(literatures):
            title, abstract = literature['title'], literature['abstract']
            item = {
                'title': title,
                'abstract': abstract,
                'triplet':[]
            }
            for i, (entity1_id, entity1_info) in enumerate(literature['entity'].items()):
                entity1_names, entity1_type = entity1_info['entity_name'], entity1_info['entity_type']
                if entity1_type not in entity_map:
                    continue
                entity1_type_hetionet = entity_map[entity1_type]
                if entity1_type_hetionet not in valid_type:
                    continue
                entity1_name = get_entity_name(entity1_names)
                message = template_summary.format(entity1_type, entity1_name, entity1_name, abstract)
                try:
                    ret_summary = request_api_palm(message)
                except:
                    continue
                for j, (entity2_id, entity2_info) in enumerate(literature['entity'].items()):
                    if i == j:
                        continue
                    entity2_names, entity2_type = entity2_info['entity_name'], entity2_info['entity_type']
                    if entity2_type not in entity_map:
                        continue
                    entity2_type_hetionet = entity_map[entity2_type]
                    if (entity1_type_hetionet, entity2_type_hetionet) not in entities2relation:
                        continue
                    time.sleep(2)
                    entity2_name = get_entity_name(entity2_names)
                    
                    entity_relation = entities2relation[(entity1_type_hetionet, entity2_type_hetionet)]
                    options, option2relation = build_options(entity_relation)
                    message = template_relation_extraction_ZeroCoT.format(ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options)
                    try:
                        ret_CoT = request_api_palm(demonstration+message)
                    except:
                        continue
                    if ret_CoT == []:
                        continue
                    message = template_relation_extraction_ZeroCoT_answer.format(ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options, ret_CoT)
                    
                    try:
                        ret_relation = request_api_palm(demonstration+message)
                    except:
                        continue
                    if ret_relation == []:
                        continue
                    
                    find, is_generated = False, False
                    for option, relation in option2relation.items():
                        if option in ret_relation or option[0] == ret_relation[0] or relation in ret_relation:
                            if relation == 'others, please specify by generating a short predicate in 5 words':
                                if '.' in ret_relation:
                                    relation = ret_relation.split('.')[1]
                                else:
                                    relation = ret_relation
                                is_generated = True
                            find = True
                            break
                    if not find:
                        is_generated = True
                        relation = ret_relation
                        print('NOT MATCH:', ret_relation, option2relation)
                    item['triplet'].append({
                        'entity1': {
                            'entity_name': entity1_names,
                            'entity_type': entity1_type_hetionet,
                            'entity_id': entity1_id
                        },
                        'entity2': {
                            'entity_name': entity2_names,
                            'entity_type': entity2_type_hetionet,
                            'entity_id': entity2_id
                        },
                        'relation': relation,
                        'is_generated': is_generated
                    })
            extracted.append(item)

        with open('../extracted/{}.json'.format(year), 'w') as f:
            f.write(json.dumps(extracted, indent=2))

if __name__ == '__main__':
    main()