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
import time
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
import pickle
sys.path.append('../')
from scripts.evaluate import evaluate_word_level
# Change this according to your set up

os.environ['TRANSFORMERS_CACHE'] = 'cache'

RESULTS_FNAME = 'results_lime_eten_complete.json'
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
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression = ' ')

def explain_instance(model, text_a, text_b):
    def predict_proba(texts):
        text_src = [text_a] * len(texts)
        to_predict = list(zip(text_src, texts))
        to_predict = list(map(list, to_predict))
        preds, _ = model.predict(to_predict)
        return np.vstack((preds, preds)).T
    
    predictions, raw_outputs = model.predict([[text_a, text_b]])
    exp = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), labels=(1, ),num_samples=1000)
    return predictions, exp.as_map()

def explain_dataset():
    results = []
    for idx in tqdm(range(1000)):
        runtimer = time.time()
        pred_score, expl = explain_instance(model, dataset['src'][idx], dataset['tgt'][idx])
        runtimer = time.time() - runtimer
        expl = expl[1]
        assert len(expl) == len(dataset['tgt'][idx].split(' '))
        feature_maps = np.zeros(len(expl))
        for k, v in expl:
            feature_maps[k] = v
        results.append(
            {
                'pred': float(pred_score),
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