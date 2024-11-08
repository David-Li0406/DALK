import os
from tqdm import tqdm
import time
import json
from api_utils import *

years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

template = '''Read the following abstract, extract the relationships between each entity.
You can choose the relation from: (covaries, interacts, regulates, resembles, downregulates, upregulates, associates, binds, treats, palliates), or generate a new predicate to describe the relationship between the two entities.
Output all the extract triples in the format of "head | relation | tail". For example: "Alzheimer's disease | associates | memory deficits"

Abstract: {}
Entity: {}
Output: '''


def read_literature():
    year2literatures = {year: [] for year in years}
    for year in years:
        with open(os.path.join('by_year_new', '{}.pubtator'.format(year))) as f:
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

def main():
    no_relation, with_relation = 0, 0
    year2literatures = read_literature()

    for year, literatures in tqdm(year2literatures.items()):
        extracted = []
        for literature in tqdm(literatures):
            time.sleep(1)
            title, abstract = literature['title'], literature['abstract']
            item = {
                'title': title,
                'abstract': abstract,
                'triplet':[]
            }
            entity_names = ', '.join([get_entity_name(entity_info['entity_name']) for entity_info in literature['entity'].values()])
            message = template.format(abstract, entity_names)
            try:
                ret = request_api_palm(message)
            except Exception as E:
                continue
            if ret == []:
                continue
            for triple in ret.split('\n'):
                if triple == '':
                    continue
                try:
                    entity1, relation, entity2 = triple.split(' | ')
                except:
                    continue
                item['triplet'].append({
                    'entity1': {
                        'entity_name': entity1,
                    },
                    'entity2': {
                        'entity_name': entity2,
                    },
                    'relation': relation,
                })
            extracted.append(item)

        with open('../extracted/{}_s2s.json'.format(year), 'w') as f:
            f.write(json.dumps(extracted, indent=2))

if __name__ == '__main__':
    main()