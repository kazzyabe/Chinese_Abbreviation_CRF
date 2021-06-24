import argparse
# from sklearn.externals import joblib
from nltk.tag import hmm
import pickle 

parser = argparse.ArgumentParser(description='Abbreviating given Chinese text')

parser.add_argument('--f', default="words.txt",
                    help='specifies input file: a word in each line')

args = parser.parse_args()
# print(args.f)
'''Load input file'''
f = open(args.f, 'r')
inp = f.read().split('\n')

# try:
#     tagger = pickle.load(open('tagger.p', 'rb'))
# except:
#     trainer = hmm.HiddenMarkovModelTrainer()
#     Tagged_Abb = pickle.load(open('Tagged_tr.p','rb'))
#     tagger = trainer.train_supervised(Tagged_Abb)
#     pickle.dump(tagger, open('tagger.p', 'wb')) 

trainer = hmm.HiddenMarkovModelTrainer()
Tagged_Abb = pickle.load(open('Tagged_tr.p','rb'))
tagger = trainer.train_supervised(Tagged_Abb)

def char_seg(t):
    tmp = []
    for c in t:
        tmp.append(c)
    return tmp

for i in inp:
    tagged = tagger.tag(char_seg(i))
    tmp = ''
    for t in tagged:
        if t[1] == 'A':
            tmp += t[0]
    print("{}: {}(abbreviated)".format(i,tmp))