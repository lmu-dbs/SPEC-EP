import numpy as np
import itertools
import math
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, root_mean_squared_error, mean_absolute_error
from concurrent.futures import ProcessPoolExecutor

import logging

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, pred: list[int], actual: list[int], attribute: str = None, exclude_nan_actual_obs: bool = True):
        self.pred = pred
        self.actual = actual
        self.attribute = attribute
        self.exclude_nan_actual_obs = exclude_nan_actual_obs

    def get_nan_share(self) -> float:
        none_share = sum([pred is None for pred in self.pred]) / len(self.pred)
        return none_share

    def _batch_seqs(self, seqs: list[list[int]], nbatches: int):
        nseqs = len(seqs)
        batchsize = math.ceil(nseqs / nbatches)
        for ndx in range(0, nseqs, batchsize):
            yield seqs[ndx : min(ndx + batchsize, nseqs)]


class NextContextEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        attr_string = f" - {self.attribute}" if self.attribute else ""
        logger.info(f"Starting evaluation - Next Attribute{attr_string}")

        if self.exclude_nan_actual_obs:
            actual_nan_idxs = [idx for idx, a in enumerate(self.actual) if not isinstance(a, str) and np.isnan(a)]
            self.pred, self.actual = (list(x) for x in zip(*[(p, a) for p, (a_idx, a) in zip(self.pred, enumerate(self.actual)) if a_idx not in actual_nan_idxs]))

    # discrete eval metrics
    def calc_accuracy_score(self) -> float:
        acc_score = accuracy_score(self.actual, self.pred)
        return acc_score
    
    def calc_balanced_accuracy_score(self) -> float:
        balanced_acc_score = balanced_accuracy_score(self.actual, self.pred)
        return balanced_acc_score

    # continuous eval metrics
    def calc_rmse(self) -> float:
        rmse = root_mean_squared_error(self.actual, self.pred)
        return rmse
    
    def calc_mae(self) -> float:
        mae = mean_absolute_error(self.actual, self.pred)
        return mae

class SFXContextEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        attr_string = f" - {self.attribute}" if self.attribute else ""
        logger.info(f"Starting evaluation - SFX{attr_string}")

        if self.exclude_nan_actual_obs:

            cropped_actuals = list()
            cropped_preds = list()
            for act_seq, pred_seq in zip(self.actual, self.pred):
                cropped_actual = act_seq[:-1]
                cropped_pred = pred_seq[:-1]
                if len(cropped_actual) > 0:
                    cropped_actuals.append(cropped_actual)
                    cropped_preds.append(cropped_pred)
            self.pred, self.actual = (cropped_preds, cropped_actuals)

    # override get_nan_share for sequences
    def get_nan_share(self) -> float:
        none_share = sum([len(pred)==0 for pred in self.pred]) / len(self.pred)
        return none_share

    # discrete eval metrics
    def calc_ndls(self, horizon: int | None = None, ncores: int = 1) -> float:
        ndls_values = list()
        if ncores == 1:
            for pred, actual in zip(self.pred, self.actual):
                try:
                    ndls_values.append(
                        normalized_damerau_levenshtein_similarity(
                            pred=pred, actual=actual, horizon=horizon
                        )
                    )
                except TypeError:
                    ndls_values.append(0)
            ndls = sum(ndls_values) / len(self.pred)
        else:
            pred_batches = self._batch_seqs(self.pred, nbatches=ncores)
            actual_batches = self._batch_seqs(self.actual, nbatches=ncores)
            with ProcessPoolExecutor(max_workers=ncores) as executor:
                batch_ndls_values = executor.map(
                    self._batch_calc_ndls,
                    pred_batches,
                    actual_batches,
                    [horizon] * ncores,
                )
            ndls_values = list(itertools.chain(*batch_ndls_values))
            ndls = sum(ndls_values) / len(ndls_values)

        return ndls

    def _batch_calc_ndls(self, pred, actual, horizon):
        ndls_values = list()
        for pred, actual in zip(pred, actual):
            try:
                ndls_values.append(
                    normalized_damerau_levenshtein_similarity(
                        pred=pred, actual=actual, horizon=horizon
                    )
                )
            except TypeError:
                ndls_values.append(0)
        return ndls_values
    
    # continuous eval metrics (evaluated at n steps into future / last predicted event)
    def calc_rmse_last(self):
        last_actuals, last_preds = self._get_last()
        rmse_last = root_mean_squared_error(last_actuals, last_preds)
        return rmse_last
    
    def calc_mae_last(self):
        last_actuals, last_preds = self._get_last()
        mae_last = mean_absolute_error(last_actuals, last_preds)
        return mae_last
    
    def calc_rmse_at_n(self, n):
        n_actuals, n_preds = self._get_n(n)
        rmse_at_n = root_mean_squared_error(n_actuals, n_preds)
        return rmse_at_n
    
    def calc_mae_at_n(self, n):
        n_actuals, n_preds = self._get_n(n)
        mae_at_n = mean_absolute_error(n_actuals, n_preds)
        return mae_at_n
    
    def calc_mae_cumsum(self, truncate_negative: bool):
        cumsum_actuals, cumsum_preds = self._calc_cumsums(truncate_negative=truncate_negative)
        mae_cumsum = mean_absolute_error(cumsum_actuals, cumsum_preds)
        return mae_cumsum

    def calc_rmse_cumsum(self, truncate_negative: bool):
        cumsum_actuals, cumsum_preds = self._calc_cumsums(truncate_negative=truncate_negative)
        rmse_cumsum = root_mean_squared_error(cumsum_actuals, cumsum_preds)
        return rmse_cumsum

    def _get_last(self):
        last_actuals = [a[-1] for a in self.actual]
        last_preds = [p[-1] if len(p)>0 else None for p in self.pred]

        last_actuals = [la for la, lp in zip(last_actuals, last_preds) if lp is not None]
        last_preds = [lp for lp in last_preds if lp is not None]

        return last_actuals, last_preds

    def _get_n(self, n):        
        n_actuals = [a[n-1] if (n-1) <= len(a) else a[-1] for a in self.actual]
        n_preds = [np for np in n_preds if np is not None]
 
        n_actuals = [na for na, np in zip(n_actuals, n_preds) if np is not None]
        n_preds = [None if len(p) == 0 else p[n-1] if (n-1) <= len(p) else p[-1] for p in self.pred]

        return n_actuals, n_preds

    def _calc_cumsums(self, truncate_negative: bool):
        cumsum_actuals = [sum(a) for a in self.actual]

        if truncate_negative:
            cumsum_preds = [sum(p) for p in self.pred if p is not None]
        else:
            cumsum_preds = [trunc_sum(p) for p in self.pred if p is not None]

        return cumsum_actuals, cumsum_preds

def damerau_levenshtein_dist(
    pred: list, actual: list, horizon: int | None = None
) -> float:
    if horizon:
        pred = [el for idx, el in enumerate(pred) if idx < horizon]
        actual = [el for idx, el in enumerate(actual) if idx < horizon]

    distance_mat = np.zeros((len(pred) + 1, len(actual) + 1))

    cp = {symbol: 0 for symbol in set(pred + actual)}

    distance_mat[:, 0] = range(0, len(pred) + 1)
    distance_mat[0, :] = range(0, len(actual) + 1)

    for i in range(0, len(pred)):

        cs = 0

        for j in range(0, len(actual)):

            if pred[i] == actual[j]:
                d = 0
            else:
                d = 1

            distance_mat[i + 1, j + 1] = min(
                distance_mat[i, j + 1] + 1,
                distance_mat[i + 1, j] + 1,
                distance_mat[i, j] + d,
            )

            i_prime = cp[actual[j]]
            j_prime = cs

            if i_prime > 0 and j_prime > 0:
                distance_mat[i + 1, j + 1] = min(
                    distance_mat[i + 1, j + 1],
                    distance_mat[int(i_prime) - 1, int(j_prime) - 1]
                    + (i + 1 - i_prime)
                    + (j + 1 - j_prime)
                    - 1,
                )

            if pred[i] == actual[j]:
                cs = j + 1

        cp[pred[i]] = i + 1

    dl_dist = distance_mat[len(pred), len(actual)]

    return dl_dist


def normalized_damerau_levenshtein_similarity(
    pred: list, actual: list, horizon: int | None = None
) -> float:
    dl_dist = damerau_levenshtein_dist(pred=pred, actual=actual, horizon=horizon)
    normalized_dist = dl_dist / max(len(pred), len(actual))
    ndls = 1 - normalized_dist

    return ndls

def trunc_sum(x):
    truncated_sum = sum([e for e in x if e >= 0])
    return truncated_sum
