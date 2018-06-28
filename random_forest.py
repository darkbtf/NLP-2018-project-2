from nltk.tokenize import word_tokenize
import numpy as np

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

VOCAB = 4902
N_TRAIN = 8000
N_TEST = 2717

f = open('TRAIN_FILE.txt', 'r')

raw_sents = []
raw_rels = []

for i in range(N_TRAIN):
    line = f.readline()[:-1]
    no, sent = line.split('\t')
    relation = f.readline()[:-1]
    f.readline()
    f.readline()
    raw_sents.append(sent)
    raw_rels.append(relation)

f = open('TEST_FILE.txt', 'r')

test_sents = []
test_rels = []

for i in range(N_TEST):
    line = f.readline()[:-1]
    no, sent = line.split('\t')
    test_sents.append(sent)

f = open('answer_key.txt', 'r')
for i in range(N_TEST):
    line = f.readline()[:-1]
    no, rel = line.split('\t')
    test_rels.append(rel)

rel_mapping = {
    'Component-Whole(e1,e2)': 0,
    'Component-Whole(e2,e1)': 1,
    'Instrument-Agency(e1,e2)': 2,
    'Instrument-Agency(e2,e1)': 3,
    'Member-Collection(e1,e2)': 4,
    'Member-Collection(e2,e1)': 5,
    'Cause-Effect(e1,e2)': 6,
    'Cause-Effect(e2,e1)': 7,
    'Entity-Destination(e1,e2)': 8,
    'Entity-Destination(e2,e1)': 9,
    'Content-Container(e1,e2)': 10,
    'Content-Container(e2,e1)': 11,
    'Message-Topic(e1,e2)': 12,
    'Message-Topic(e2,e1)': 13,
    'Product-Producer(e1,e2)': 14,
    'Product-Producer(e2,e1)': 15,
    'Entity-Origin(e1,e2)': 16,
    'Entity-Origin(e2,e1)': 17,
    'Other': 18
}
names = list(map(lambda x: x[0], sorted(rel_mapping.items(), key=lambda x: x[1])))


def to_category(rel):
    return rel_mapping[rel]
def to_name(num):
    return names[num]


token_vec = []

vocabs = {}

def to_tokens(sents, train=True):
    features = []
    for i in range(len(sents)):
        raw_toks = word_tokenize(sents[i])
        feats = [[] for j in range(3)]
        k = 0
        col, enc = False, False
        for tok in raw_toks:
            if col:
                if tok == '>':
                    col = False
                    enc = not enc
                    k += 1
                else: continue

            if tok == '<': col = True
            elif not enc and tok.isalnum():
                feats[k//2].append(stemmer.stem(tok))
                if train:
                    if tok not in vocabs: vocabs[tok] = 0
                    vocabs[tok] += 1
        features.append(feats)
    return features

token_vec = to_tokens(raw_sents)
test_token_vec = to_tokens(test_sents, train=False)

ls = sorted(list(vocabs.items()), key=lambda x: x[1], reverse=True)[:VOCAB]
key = 1
word_index = {}
for k, v in ls:
    word_index[k] = key
    key += 1

def bow(toks):
    inds = list(map(lambda tok: 0 if tok not in word_index else word_index[tok], toks))
    vec = [0] * (VOCAB + 1)
    for ind in inds: vec[ind] += 1
    return vec

train_X = []
train_Y = []

for i in range(8000):
    train_X.append(bow(token_vec[i][0])+bow(token_vec[i][1])+bow(token_vec[i][2]))
    train_Y.append(to_category(raw_rels[i]))

train_X = np.array(train_X)

test_X = []
test_Y = []
for i in range(N_TEST):
    test_X.append(bow(test_token_vec[i][0])+bow(test_token_vec[i][1])+bow(test_token_vec[i][2]))
    test_Y.append(to_category(test_rels[i]))

test_X = np.array(test_X)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, max_features='auto', random_state=0, oob_score=True)

clf.fit(train_X, train_Y)

f = open('rf_answer.txt', 'w')
k = 8001

for i in range(len(test_result)):
    f.write(str(k) + '\t' + to_name(test_result[i]) + '\n')
    k += 1
f.close()