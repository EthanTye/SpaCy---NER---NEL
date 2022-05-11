import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy.tokens import DocBin, Doc, Span
import json
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import numpy as np
from spacy.kb import KnowledgeBase
import pandas as pd
spacy.prefer_gpu(gpu_id=0)

def load_data_people(file_path):
    data = pd.read_pickle(file_path)
    return(data)

#def load_data_org(file_org)

def save_data(file_path, data):
    with open(file_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, indent = 4)

def knowledge_base_people(file_path_p, file_path_o):
    nlp_lg = spacy.load("en_core_web_lg")
    kb = KnowledgeBase(vocab=nlp_lg.vocab, entity_vector_length=300)
    data = load_data_people(file_path_p)
    for index, row in data.iterrows():
        desc_doc = nlp_lg(row.description_full)
        desc_enc = desc_doc.vector
        #if row.appointment != 'Backbencher':
        kb.add_entity(entity=row.mp, entity_vector =desc_enc, freq=42)
        kb.add_alias(alias=row.mp, entities=[row.mp], probabilities=[1])
        kb.add_entity(entity = 'Sylvia Chan',
                      entity_vector = nlp_lg('NOC Co-Founder Sylvia Chan Was Diagnosed With Severe Depression, OCD & Rage Disorder When She Was In JC (unapologetic and sue all your ass syndrome too)').vector,
                      freq = 30)
        kb.add_alias(alias='Sylvia Chan', entities=['Sylvia Chan'], probabilities=[1])
        try:
            if row.initials != 'NS' and len(row.initials) != 0:
                kb.add_alias(alias=row.initials, entities=[row.mp], probabilities=[1])
            if row.given_names != 'Desmond' and row.given_names !='Louis' and row.given_names !='Sylvia':
                kb.add_alias(alias=row.given_names, entities=[row.mp], probabilities=[1])
            for alias in row.aliases:
                if alias != '':
                    kb.add_alias(alias=alias, entities=[row.mp], probabilities=[1])
        except: print('error')
    kb.add_alias(alias='Desmond', entities=["Desmond Choo", "Desmond Lee", 'Desmond Tan'], probabilities=[0.001, 0.599, 0.4]) #probability that the name Desmond belongs to either 1 of 3 minister/mayors named Desmond. Desmond Lee is most frequently mentioned followed by Desmond Tan
    kb.add_alias(alias='Louis', entities=['Louis Chua', 'Louis Ng'], probabilities=[0.5, 0.5]) # again Louis can refer to either 1 of 2 prominent Mps so we'll add probabilities here
    kb.add_alias(alias='Slyvia', entities=['Sylvia Lim', 'Sylvia Chan'], probabilities=[0.6, 0.4])
    for item in data.ward.unique():
        kb.add_alias(alias = f'Member of {item}', entities = [ent for ent in data[data.ward == item]['mp']], probabilities = [1/len(data[data.ward == item]) for ent in data[data.ward == item]['mp']])

    data_o = load_data_people(file_path_o)
    for index, row in data_o.iterrows():
        desc_doc_o = nlp_lg(row.description)
        desc_enc_o = desc_doc_o.vector
        kb.add_entity(entity = row.org, entity_vector = desc_enc_o, freq = 42)
        kb.add_alias(alias = row.org, entities = [row.org], probabilities = [1])
        for alias in row.aliases:
            if alias != '':
                kb.add_alias(alias = alias, entities = [row.org], probabilities = [1])
    kb.to_disk('../assets/mp_kb')
    nlp_lg.vocab.to_disk('mp_vocab')

knowledge_base_people('../data/mplist_clean.pk1', '../data/orglist_clean.pk1')


def generate_patterns(file, file_o):
    data = load_data_people(file)
    name_pattern = []
    for index, row in data.iterrows():
        name_pattern.append({'label': 'PERSON', 'pattern': row.mp, 'id': row.mp})
        #name_pattern.append({'label': 's_pol','pattern': f'mp of {row.ward}', 'id': row.mp })
        if row.appointment != 'Backbencher':
            given_name_pattern = {'label': 'PERSON', 'pattern': row.given_names, 'id': row.mp}
            name_pattern.append(given_name_pattern)
            if row.initials != 'NS':
                initials_pattern = {'label': 'PERSON', 'pattern': row.initials.upper(), 'id': row.mp}
                name_pattern.append(initials_pattern)
        for alias in row.aliases:
            if alias != '':
                alias_pattern = {'label': 'PERSON', 'pattern': alias, 'id': row.mp}
                name_pattern.append(alias_pattern)
    data_o = load_data_people(file_o)
    for index, row in data_o.iterrows():
        name_pattern.append({'label': 'ORG', 'pattern': row.org, 'id': row.org})
        for alias in row.aliases:
            alias_pattern = {'label': 'ORG', 'pattern': alias, 'id': row.org}
            name_pattern.append(alias_pattern)
    return(name_pattern)
        #name_pattern.append({'label': 'ORG', 'pattern': row.party, 'id': row.qid})
        #name_pattern.append({'label': 'ORG', 'pattern': "".join([i[0] for i in row.party.split(' ')]), 'id': row.qid})




    #ruler.add_patterns(org_pattern)
    #target_ent_ids.append(row.mp)
    #target_ent_ids.append(row.party)
    #print(name_pattern)
    #print(name_pattern)
    #test_doc = nlp(f'{row.initials} or {row.mp} is a {row.party} member and in the {row.ward} and the {row.appointment}')
    #print([(ent.text, ent.label_, ent.ent_id_) for ent in test_doc.ents])


def generate_model(name_pattern):
    nlp = spacy.load('en_core_web_trf')
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(name_pattern)
    #nlp.remove_pipe('ner')
    #senter = nlp.add_pipe('sentencizer')
    nlp.to_disk('../assets/mp_ner')


def nlp_model (model, text_list):
    db = DocBin( attrs = ["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"])
    for doc in tqdm(model.pipe(text_list, disable = ["lemmatizer", "attribute_ruler", "ner"])):
        if len(doc.ents)>0:
            db.add(doc)

    return (db)



    #db = DocBin(docs = docs, attrs = ["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"])
    '''
    for doc in tqdm(model.pipe(text_list)):
        entities = []
        for ent in doc.ents:
            span = doc.char_span(ent.start_char, ent.end_char, ent.label_, ent.kb_id_)
            entities.append(span)
        if len(entities)>0:
            doc.ents = entities
            db.add(doc)
 
    '''
    return(db)


def nlp_model_el (model, text):
    doc = model(text)
    results = []
    entities = []
    for ent in doc.ents:
        entities.append([ent.start_char, ent.end_char, ent.label_, ent.ent_id_, 1.0])
    if len(entities)> 0:
        results = [doc.text, {'links': entities}]
    return(results)




patterns = generate_patterns('../data/mplist_clean.pk1','../data/orglist_clean.pk1')
#print(patterns)
generate_model(patterns)

nlp = spacy.load('../assets/mp_ner')

df = pd.read_csv('../data/Comments_train3.csv')
df2 = pd.read_csv('../data/100k_subs.csv')
df.head()
df.dropna(subset = ['comment'], axis = 0, inplace = True)
df2.dropna(subset = ['title'], axis = 0, inplace = True)
df['comment_len'] = [len(comment) for comment in df.comment]
df2['title_len'] = [len(title) for title in df2.title]
df = df[df.comment_len<=512]
df = df[df.comment_len>=10]
df2 = df2[df2.title_len<=512]
df2 = df2[df2.title_len>=10]

titles = [title for title in df2.title]
comments = [comment for comment in df.comment]
user = [user for user in df.user]

for item in titles: # creating a list of training items that is made up of a mix of comments + submission titles
    comments.append(item)
    user.append(item)

# using train text split to split up the texts into a test and val set
train_comments, val_comments, _, _ = train_test_split(comments, user, train_size = 0.8, random_state = 42)



def data_creator_el(comment_list):
    data = []
    hits = []
    for comment in comment_list:
        results = nlp_model_el(nlp,comment)
        hits.append(results)
        if results != None:
            data.append(results)
        elif np.random.choice([0,1], p = [0.99, 0.1]) == 1:
            data.append(results)
    return(data)



#data_full = []
#def data_creator(comment_list):
    #data = []
    #hits = []
    #results = nlp_model(nlp,comment_list)
    #return(results)



def data_converter(dataset):
    db = DocBin()
    for text, annot in tqdm(dataset):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot['entities']:
            span = doc.char_span(start, end, label=label, alignment_mode = 'contract')
            if span is None:
                pass
            else: ents.append(span)
        doc.ents = ents
        db.add(doc)
    return(db)

TRAIN_DATA = nlp_model(nlp, train_comments)
VAL_DATA = nlp_model(nlp, val_comments)

TRAIN_DATA.to_disk('../data/mp_train2.spacy')
VAL_DATA.to_disk('../data/mp_val2.spacy')

#TRAIN_DATA_EL = data_creator_el(train_comments)
#VAL_DATA_EL = data_creator_el(val_comments)


#save_data('../data/mp_train_el.json', TRAIN_DATA_EL)
#save_data('../data/mp_val_el.json', VAL_DATA_EL)

