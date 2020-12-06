import argparse
import pickle 
import sklearn_crfsuite
from sklearn_crfsuite import metrics

parser = argparse.ArgumentParser(description='Abbreviating given Chinese text')

parser.add_argument('-f', default="words.txt",
                    help='specifies input file: a word in each line')

args = parser.parse_args()

def char2features(word, i):
    '''
    word: whole word
    i: index of character
    
    return features
        where features is a dictionary containing:
            char: target character
    '''
    features = {
        'bias': 1.0,
        'char': word[i],
        'i': i,
        'word length': len(word)
    }
    
    # previous char info
    if i > 0:
        features['Prev'] = word[i-1]
        if i > 1:
            features['Prev2'] = word[i-2]
        else:
            features['Prev2'] = "None"
    else:
        features.update({
            'Prev': "None",
            'Prev2': "None"
        })
    
    # post char info
    if i < len(word)-1:
        features['Post'] = word[-1]
        if i < len(word)-2:
            features['Post2'] = word[-2]
        else:
            features['Post2'] = "None"
    else:
        features.update({
            'Post': "None",
            'Post2': "None"
        })
        
    return features

def word2features(word):
    return [char2features(word, i) for i in range(len(word))]

if __name__ == "__main__":
    x_train = pickle.load(open("x_train_crf.p", "rb"))
    y_train = pickle.load(open("y_train_crf.p", "rb"))

    ## training

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)

    if args.f == 'test':
        labels = list(crf.classes_)
        x_test = pickle.load(open("x_test_crf.p", "rb"))
        y_test = pickle.load(open("y_test_crf.p", "rb"))
        y_pred = crf.predict(x_test)
        res = metrics.flat_f1_score(y_test, y_pred,
                        average='weighted', labels=labels)
        print(res)
    else:
        '''Load input file'''
        f = open(args.f, 'r')
        inp = f.read().split('\n')
        x = [word2features(word) for word in inp]

        y_pred = crf.predict(x)
        print(y_pred)

        for i in range(len(inp)):
            tmp = ""
            for j in range(len(inp[i])):
                if y_pred[i][j] == 'A':
                    tmp += inp[i][j]
            print(tmp)


    