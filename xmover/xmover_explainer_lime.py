import sys
sys.path.append('./xmover')
import pandas as pd
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
import os
from mosestokenizer import MosesDetokenizer, MosesTokenizer
import argparse
from scorer import XMOVERScorer
from tqdm import tqdm
import time
import json
import numpy as np
import torch
import truecase

class XMoverWrapper():
    def __init__(self, src_lang, tgt_lang, model_name, do_lower_case, language_model, mapping, device, ngram, bs):
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang
        self.mapping = mapping
        self.device = device
        self.ngram = ngram
        self.bs = bs

        temp = np.loadtxt('europarl-v7.' + src_lang + '-' + tgt_lang + '.2k.12.BAM.map')
        self.projection = torch.tensor(temp, dtype=torch.float).to(device)
        
        temp = np.loadtxt('europarl-v7.' + src_lang + '-' + tgt_lang + '.2k.12.GBDD.map')
        self.bias = torch.tensor(temp, dtype=torch.float).to(device)

        self.scorer = XMOVERScorer(model_name, language_model, do_lower_case, device)
        self.rows = 0

        self.src_sent = None

    def __call__(self, translations):
        assert self.src_sent is not None
        source = [self.src_sent] * len(translations)
        xmoverscores = self.scorer.compute_xmoverscore(self.mapping, self.projection, self.bias, source, translations, self.ngram, self.bs)
        preds = np.array(xmoverscores) 
        results = np.vstack((preds, preds)).T
        return results

    def tokenize_sent(self, sentence, lang):
        # with MosesTokenizer(lang) as tokenize:        
            # tokens = tokenize(sentence)
        # return tokens       
        return sentence.split()

    def detokenize(self, tokens, lang):
        # with MosesDetokenizer(lang) as detokenize:        
            # sent = detokenize(tokens)
        # return sent 
        return ' '.join(tokens)

    def build_feature(self, trans_sent):
        tokens = self.tokenize_sent(trans_sent, self.tgt_lang)
        tdict = {}
        for i,tt in enumerate(tokens):
            tdict['{}_{}'.format(tt,i)] = tt

        df = pd.DataFrame(tdict, index=[0])
        return df

    def mask_model(self, mask, x):
        tokens = []
        for mm, tt in zip(mask, x):
            if mm: tokens.append(tt)
            else: tokens.append('[MASK]')
        trans_sent = self.detokenize(tokens, self.tgt_lang)
        sentence = pd.DataFrame([trans_sent])
        return sentence

class ExplainableXMover():
    def __init__(self, src_lang, tgt_lang, model_name='bert-base-multilingual-cased', do_lower_case=False, language_model='gpt2', mapping='CLP', device='cuda:0', ngram=2, bs=32):
        self.wrapper = XMoverWrapper(src_lang, tgt_lang, model_name, do_lower_case, language_model, mapping, device, ngram, bs)
        

    def __call__(self, src_sent, trans_sent):
        #return self.wrapper.scorer.compute_xmoverscore(self.wrapper.mapping, self.wrapper.projection, self.wrapper.bias, [self.wrapper.detokenize(src_sent.split(),self.wrapper.src_lang)], [self.wrapper.detokenize(trans_sent.split(),self.wrapper.tgt_lang)], self.wrapper.ngram, self.wrapper.bs)[0]
        self.wrapper.src_sent = src_sent
        preds = self.wrapper(trans_sent)
        return preds

    # def explain(self, src_sent, trans_sent, plot=False):
    #     self.wrapper.src_sent = src_sent
    #     value = self.explainer(self.wrapper.build_feature(trans_sent))
    #     if plot: shap.waterfall_plot(value[0])
    #     all_tokens = self.wrapper.tokenize_sent(trans_sent, self.wrapper.tgt_lang)

    #     return [(token,sv) for token, sv in zip(all_tokens,value[0].values)]

def explain_instance(model, explainer, text_a, text_b):
    def predict_proba(texts):
        return model(text_a, texts)

    #predictions, raw_outputs = model.predict([[text_a, text_b]])
    exp = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), labels=(1, ),num_samples=1000)
    return exp.as_map()

RESULTS_FNAME = 'results_lime_xmover_roen.json'
SRC_LANG = 'ro'
TGT_LANG = 'en'
SPLIT = 'dev'

data_dir = f'../../data/{SPLIT}/{SRC_LANG}-{TGT_LANG}-{SPLIT}'
src = [s.strip() for s in open(f'{data_dir}/{SPLIT}.src').readlines()]
tgt = [s.strip() for s in open(f'{data_dir}/{SPLIT}.mt').readlines()]
wor = [list(map(int, s.strip().split())) for s in open(f'{data_dir}/{SPLIT}.tgt-tags').readlines()]
sen = [float(s.strip()) for s in open(f'{data_dir}/{SPLIT}.da').readlines()]
assert len(src) == len(tgt) == len(wor) == len(sen)
dataset = {'src': src, 'tgt': tgt, 'word_labels': wor, 'sent_labels': sen}

if __name__ == '__main__':
    model = ExplainableXMover('et','en')
    explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression = ' ')
    src = 'Kant ütleb , et kõik käsitööd , tööndused ja kunstid on tööjaotusest võitnud ning filosoofia peaks neist eeskuju võtma .'
    trans = 'Kant says that all crafts , all works and all the arts have gained from the division of labour , and the philosophy should take inspiration from them .'
    #score = model(src, trans)
    def explain_dataset():
        results = []
        for idx in tqdm(range(1000)):
            runtimer = time.time()
            expl = explain_instance(model,explainer, dataset['src'][idx], dataset['tgt'][idx])
            runtimer = time.time() - runtimer
            expl = expl[1]
            assert len(expl) == len(dataset['tgt'][idx].split(' '))
            feature_maps = np.zeros(len(expl))
            for k, v in expl:
                feature_maps[k] = v
            results.append(
                {
                    'expl': list(feature_maps),
                    'ground_truth_word': dataset['word_labels'][idx],
                    'ground_truth_sent': dataset['sent_labels'][idx],
                    'time': runtimer
                }
            )
            print(runtimer)
        json.dump(results, open(RESULTS_FNAME, 'w'))
        return results      

    results = explain_dataset()
    
    