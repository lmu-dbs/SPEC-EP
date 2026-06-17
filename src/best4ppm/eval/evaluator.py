import numpy as np
import itertools
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor

import logging
logger = logging.getLogger(__name__)

class Evaluator:

    def __init__(self, pred: list[int], actual: list[int], split_context: bool = False, act_encoder: LabelEncoder = None, start_token: str = 'START', end_token: str = 'END', unseen_token: str = 'NONE/UNSEEN'):        
        self.pred = pred
        self.actual = actual
        self.split_context = split_context
        self.act_encoder = act_encoder
        self.start_token = start_token
        self.end_token = end_token
        self.unseen_token = unseen_token
    
    def get_nan_share(self) -> float:
        
        none_share = sum([pred==self.unseen_token for pred in self.pred_activity])/len(self.pred_activity)
        return none_share
    
    def _batch_seqs(self, seqs: list[list[int]], nbatches: int):
        nseqs = len(seqs)
        batchsize = math.ceil(nseqs/nbatches)
        for ndx in range(0, nseqs, batchsize):
            yield seqs[ndx:min(ndx + batchsize, nseqs)]

class NAPEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('Starting evaluation - NAP')

        if self.unseen_token not in self.act_encoder.classes_:
            unseen_act_idx = len(self.act_encoder.classes_)
            self.act_encoder.classes_ = np.append(self.act_encoder.classes_, self.unseen_token)
        else:
            unseen_act_idx = list(self.act_encoder.classes_).index(self.unseen_token)
        self.actual = [a if a != -1 else unseen_act_idx for a in self.actual]
        self.no_none_pred = [p if p is not None else unseen_act_idx for p in self.pred]

        if self.split_context:
            # decode predicted and actual activities for enabling splitting activity and context_cluster
            pred_decoded = self.act_encoder.inverse_transform(self.no_none_pred)
            actual_decoded = self.act_encoder.inverse_transform(self.actual)

            # splitting the strings
            pred_splits = [split_activity_and_context(p, '_context_cluster_', self.start_token, self.end_token, self.unseen_token) for p in pred_decoded]
            actual_splits = [split_activity_and_context(a, '_context_cluster_', self.start_token, self.end_token, self.unseen_token) for a in actual_decoded]
            self.pred_activity, self.pred_context = zip(*pred_splits)
            self.actual_activity, self.actual_context = zip(*actual_splits)

        else:
            self.pred_activity = self.no_none_pred
            self.pred_context = None
            self.actual_activity = self.actual
            self.actual_context = None

    def calc_accuracy_score(self) -> float:
        acc_score = accuracy_score(self.actual_activity, self.pred_activity)
        return acc_score

    def calc_balanced_accuracy_score(self) -> float:
        balanced_acc_score = balanced_accuracy_score(self.actual_activity, self.pred_activity)
        return balanced_acc_score

class SFXEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('Starting evaluation - RTP')

        if self.unseen_token not in self.act_encoder.classes_:
            unseen_act_idx = len(self.act_encoder.classes_)
            self.act_encoder.classes_ = np.append(self.act_encoder.classes_, self.unseen_token)
        else:
            unseen_act_idx = list(self.act_encoder.classes_).index(self.unseen_token)
        self.actual = [[a if a != -1 else unseen_act_idx for a in seq] for seq in self.actual]
        
        self.no_none_pred = [[p if p is not None else unseen_act_idx for p in seq] for seq in self.pred]

        if self.split_context:
            # decode predicted and actual sequences for enabling splitting activity and context_cluster

            pred_decoded = [self.act_encoder.inverse_transform(seq) for seq in self.no_none_pred]
            actual_decoded = [self.act_encoder.inverse_transform(seq) for seq in self.actual]

            # then split the strings
            pred_splits = [[split_activity_and_context(p, '_context_cluster_', self.start_token, self.end_token, self.unseen_token) for p in seq] for seq in pred_decoded]
            actual_splits = [[split_activity_and_context(a, '_context_cluster_', self.start_token, self.end_token, self.unseen_token) for a in seq] for seq in actual_decoded]
            pred_acts = list()
            pred_contexts = list()
            actual_acts = list()
            actual_contexts = list()
            for pred_split, actual_split in zip(pred_splits, actual_splits):
                pred_act, pred_context = zip(*list(pred_split)) if len(pred_split) != 0 else ([self.unseen_token],[-1])
                actual_act, actual_context = zip(*actual_split) if len(actual_split) != 0 else ([self.unseen_token],[-1])
                pred_acts.append(list(pred_act))
                pred_contexts.append(list(pred_context))
                actual_acts.append(list(actual_act))
                actual_contexts.append(list(actual_context))
            
            self.pred_activity, self.pred_context = (pred_acts, pred_contexts)
            self.actual_activity, self.actual_context = (actual_acts, actual_contexts)

        else:
            self.pred_activity = self.no_none_pred
            self.pred_context = None
            self.actual_activity = self.actual
            self.actual_context = None

    def calc_ndls(self, horizon: int|None = None, ncores: int = 1) -> float:
        ndls_values = list()
        if ncores==1:
            for pred, actual in zip(self.pred_activity, self.actual_activity):
                try:
                    ndls_values.append(normalized_damerau_levenshtein_similarity(pred=pred, actual=actual, horizon=horizon))
                except TypeError:
                    ndls_values.append(0)
            ndls = sum(ndls_values)/len(self.pred_activity)
        else:
            pred_batches = self._batch_seqs(self.pred_activity, nbatches=ncores)
            actual_batches = self._batch_seqs(self.actual_activity, nbatches=ncores)
            with ProcessPoolExecutor(max_workers=ncores) as executor:
                batch_ndls_values = executor.map(self._batch_calc_ndls,
                                                    pred_batches,
                                                    actual_batches,
                                                    [horizon]*ncores)
            ndls_values = list(itertools.chain(*batch_ndls_values))
            ndls = sum(ndls_values)/len(ndls_values)

        return ndls
    
    def _batch_calc_ndls(self, pred, actual, horizon):
        ndls_values = list()
        for pred, actual in zip(pred, actual):
            try:
                ndls_values.append(normalized_damerau_levenshtein_similarity(pred=pred, actual=actual, horizon=horizon))
            except TypeError:
                ndls_values.append(0)
        return ndls_values

def damerau_levenshtein_dist(pred: list, actual: list, horizon: int|None = None) -> float:
    if horizon:
        pred = [el for idx, el in enumerate(pred) if idx < horizon]
        actual = [el for idx, el in enumerate(actual) if idx < horizon]
    
    distance_mat = np.zeros((len(pred)+1, len(actual)+1))

    cp = {symbol:0 for symbol in set(pred + actual)}

    distance_mat[:,0] = range(0, len(pred)+1)
    distance_mat[0,:] = range(0, len(actual)+1)

    for i in range(0, len(pred)):
        
        cs = 0
        
        for j in range(0, len(actual)):

            if pred[i]==actual[j]:
                d = 0
            else:
                d = 1
            
            distance_mat[i+1, j+1] = min(distance_mat[i,j+1] + 1, distance_mat[i+1,j] + 1, distance_mat[i,j] + d)

            i_prime = cp[actual[j]]
            j_prime = cs

            if i_prime > 0 and j_prime > 0:
                distance_mat[i+1, j+1] = min(distance_mat[i+1,j+1], distance_mat[int(i_prime)-1, int(j_prime)-1]+(i+1-i_prime)+(j+1-j_prime)-1)
            
            if pred[i]==actual[j]:
                cs = j+1
        
        cp[pred[i]] = i+1

    dl_dist = distance_mat[len(pred), len(actual)]

    return dl_dist

def normalized_damerau_levenshtein_similarity(pred: list, actual: list, horizon: int|None = None) -> float:
    dl_dist = damerau_levenshtein_dist(pred=pred, actual=actual, horizon=horizon)
    normalized_dist = dl_dist/max(len(pred), len(actual))
    ndls = 1-normalized_dist

    return ndls

def split_activity_and_context(activity_context: str, split_str: str, start_token: str = 'START', end_token: str = 'END', unseen_token: str = 'NONE/UNSEEN'):

    if len(activity_context) == 0:
        pass
    if activity_context not in [unseen_token, start_token, end_token]:
        split_list = activity_context.split(split_str)
        activity = split_str.join(split_list[:-1])
        context = int(split_list[-1])

        if len(split_list[0]) == 0: # split_str not found in activity_context
            return None, activity
    else:
        return activity_context, -1

    return activity, context