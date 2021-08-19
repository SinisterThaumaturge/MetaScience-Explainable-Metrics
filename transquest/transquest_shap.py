import os
import sys
import numpy as np
import json
from tqdm import tqdm
from scipy.stats import pearsonr
from IPython.core.display import display, HTML
import shap
import pandas as pd
import torch
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
import pickle
sys.path.append('../')
from scripts.evaluate import evaluate_word_level
# Change this according to your set up

os.environ['TRANSFORMERS_CACHE'] = 'cache'

RESULTS_FNAME = 'results.json'
SRC_LANG = 'et'
TGT_LANG = 'en'
SPLIT = 'dev'


data_dir = f'../data/{SPLIT}/{SRC_LANG}-{TGT_LANG}-{SPLIT}'
src = [s.strip() for s in open(f'{data_dir}/{SPLIT}.src').readlines()]
tgt = [s.strip() for s in open(f'{data_dir}/{SPLIT}.mt').readlines()]
wor = [list(map(int, s.strip().split())) for s in open(f'{data_dir}/{SPLIT}.tgt-tags').readlines()]
sen = [float(s.strip()) for s in open(f'{data_dir}/{SPLIT}.da').readlines()]
assert len(src) == len(tgt) == len(wor) == len(sen)
dataset = {'src': src, 'tgt': tgt, 'word_labels': wor, 'sent_labels': sen}

# Load model
model = MonoTransQuestModel(
    'xlmroberta',
    f'TransQuest/monotransquest-da-{SRC_LANG}_{TGT_LANG}-wiki', num_labels=1, use_cuda=torch.cuda.is_available()
)

#Rewritten Explainable Wrapper for Transquest instead of XMoverScore
class ExplainableWrapper:

   def predict_proba(self,translations):
    translations = [s[0] for s in translations]
    text_src = [self.src_sent] * len(translations)
    to_predict = list(zip(text_src, translations))
    to_predict = list(map(list, to_predict))
    self.to_predict = to_predict
    preds, _ = self.model.predict(to_predict)
    self.rows += 1
    print(self.rows)
    return np.array(preds)

   def mask_model(self, mask, x):
    tokens = []
    for mm, tt in zip(mask,x):
      if mm: tokens.append(tt)
      else: tokens.append('[MASK]')
    trans_sent = ' '.join(tokens)
    sentence = pd.DataFrame([trans_sent])
    return sentence

   def build_feature(self, trans_sent):
    tokens = trans_sent.split()
    tdict = {}
    for i,tt in enumerate(tokens):
      tdict['{}_{}'.format(tt,i)] = tt
    df = pd.DataFrame(tdict, index=[0])
    return df

   def __init__(self, model):
    self.model = model
    self.explainer = shap.Explainer(self.predict_proba, self.mask_model)
    self.src_sent = None
    self.rows = 0

   def explain(self, src_sent, trans_sent, plot=False):
    self.src_sent = src_sent
    value = self.explainer(self.build_feature(trans_sent))
    if plot: shap.waterfall_plot(value[0])
    all_tokens = trans_sent.split()

    return [(token,sv) for token, sv in zip(all_tokens,value[0].values)]


explain_model = ExplainableWrapper(model)
exps = []
for i in tqdm(range(50)):
    exp = explain_model.explain(dataset['src'][i], dataset['tgt'][i])
    exps.append(exp)

with open('et_en_transquest.pkl','wb') as ff:
    pickle.dump(exps, ff)

exp_scores = []
for exp in exps:
    scores = [-entry[1] for entry in exp] # use negative SHAP values to find the incorrect tokens
    exp_scores.append(scores)

evaluate_word_level(dataset['word_labels'], exp_scores)