import json
from tqdm import tqdm
import os
import pandas as pd

# extract triplet from hetionet
hetionet = json.load(open('../Hetionet/hetionet-v1.0.json'))

id2name = {}
for node in hetionet['nodes']:
    id2name[node['identifier']] = node['name']

all_data = set()
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
for year in years:
    data = set()
    augmented = json.load(open(os.path.join('..','extracted', '{}_s2s.json'.format(year))))
    for literature in tqdm(augmented):
        for triplet in literature['triplet']:
            entity1_list, entity2_list = triplet['entity1']['entity_name'], triplet['entity2']['entity_name']
            relation = triplet['relation']
            if relation == 'no-relation':
                continue
            if (entity1_list, relation, entity2_list) not in all_data:
                data.add((entity1_list, relation, entity2_list))
                all_data.add((entity1_list, relation, entity2_list))

    data = {
        'head': [item[0] for item in data],
        'relation': [item[1] for item in data],
        'tail': [item[2] for item in data],
    }

    data = pd.DataFrame(data)
    data.to_csv('extracted_triplet_{}.txt'.format(year), header=False, index=False, sep='\t')