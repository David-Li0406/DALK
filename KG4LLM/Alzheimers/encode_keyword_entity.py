import os
import json
import pandas as pd
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

df = pd.read_csv('train_s2s.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
entity = set()
for _, item in tqdm(df.iterrows()):
    entity.add(item[0])
    entity.add(item[2])

entity=list(entity)

keyword = set()
for file in os.listdir('result_ner'):
    dataset = json.load(open(os.path.join('result_ner' ,file)))
    for item in dataset:
        k_list = item['entity'].split('\n')
        for k in k_list:
            try:
                k = k.split('.')[1].strip()
                keyword.add(k)
            except:
                print(k)
                continue

keyword = list(keyword)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.to("cuda")

# encode entities
embeddings = model.encode(entity, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entity,
    "embeddings": embeddings,
}
import pickle
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

# encode keywords
embeddings = model.encode(keyword, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keyword,
    "embeddings": embeddings,
}
import pickle
with open("keyword_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("done!")
