from llm2kg_s2s import read_literature
import json

year2literatures = read_literature()
print({k:len(v) for k, v in year2literatures.items()})
total_literatures = sum([len(v) for v in year2literatures.values()])

generative_relation, pair_relation = set(), set()
generative_triples, pair_triples = set(), set()
generative_node, pair_node = set(), set()

for year in range(2011,2021):
    kg_generative = json.load(open('../extracted/{}_s2s.json'.format(year)))
    for item_literature in kg_generative:
        for item in item_literature['triplet']:
            generative_relation.add(item['relation'])
            generative_triples.add((item['entity1']['entity_name'],item['entity2']['entity_name'],item['relation']))
            generative_node.add(item['entity1']['entity_name'])
            generative_node.add(item['entity2']['entity_name'])

    kg_pair = json.load(open('../extracted/{}_v2.json'.format(year)))
    for item_literature in kg_pair:
        for item in item_literature['triplet']:
            pair_relation.add(item['relation'])
            pair_triples.add((item['entity1']['entity_name'][0],item['entity2']['entity_name'][0],item['relation']))
            for entity in item['entity1']['entity_name']:
                pair_node.add(entity)
            for entity in item['entity2']['entity_name']:
                pair_node.add(entity)

print(total_literatures)
print('generative')
print('entity:', len(generative_node))
print('relation:', len(generative_relation))
print('triples:', len(generative_triples))


print('pair-wised')
print('entity:', len(pair_node))
print('relation:', len(pair_relation))
print('triples:', len(pair_triples))