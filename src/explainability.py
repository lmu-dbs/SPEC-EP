import time
import os
import numpy as np
import random
import pandas as pd
from itertools import chain
from collections import Counter

from best4ppm.data.sequencedata import SequenceData
from best4ppm.models.best import BESTPredictor

from contextppm.dataset.ECDataset import ECDataset
from contextppm.util.config_utils import read_config
from contextppm.clustering.ProcessContextClustering import ProcessContextClustering
from contextppm.encoding.util import Decoding

from contextppm.util.logging import init_logging
logger = init_logging(__name__, 'main.log')

from util.paths import CONFIG_PATH, DATA_PATH, EXPORT_PATH
from util.combinations import param_combinations

def main():

    general_config = read_config(os.path.join(CONFIG_PATH, "general_config.yml"))
    data_configs = read_config(os.path.join(CONFIG_PATH, "data_configs.yml"))
    model_configs = read_config(os.path.join(CONFIG_PATH, "model_configs.yml"))
    FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')

    os.makedirs(FIG_EXP_DIR, exist_ok=True)

    for dataset in general_config["dataset"]:

        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f"desired datset {dataset} not found in data_config.yml",)
            raise

        model_config = model_configs[general_config['model_config']]
        model_config_train = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_eval'}
        model_config_eval = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_train'}
        combinations_generator = param_combinations(model_config)
        config_combinations = [c_comb for c_comb in combinations_generator]
        additional_params = dict()
        
        cv_hashes = [random.getrandbits(128) for _ in range(0, len(config_combinations))]
        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset

        max_pattern_size_eval = max(model_config_eval['max_pattern_size_eval'][0])

        if max_pattern_size_eval > model_config_train['max_pattern_size_train'][0]:
            raise ValueError('max_pattern_size_train must be higher than maximum max_pattern_size_eval!')

        for comb_idx, model_params in enumerate(config_combinations):

            random.seed(additional_params['seed'])
            np.random.seed(additional_params['seed'])

            times = dict()
            times["start_time"] = time.perf_counter()
            times["data_prep_time"] = time.perf_counter()

            if not data_config.get('read_params'):
                data_config['read_params'] = dict()

            data = ECDataset.from_csv(
                load_path=os.path.join(DATA_PATH, data_config["file_name"]),
                case_identifier=data_config["case_identifier"],
                activity_identifier=data_config["activity_identifier"],
                timestamp_identifier=data_config["timestamp_identifier"],
                read_params=data_config.get('read_params'),
            )

            if general_config['cv_folds'] > 1:

                folds = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()
                base_cv_hash = cv_hashes[comb_idx]

                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                fold_models = list()

                for fold in folds:

                    data_train, data_test = fold
                    times['run_start_time'] = time.perf_counter()

                    model_params.update({'encoding_params': data_config['encoding_params'],
                                         'transform_params': data_config['transform_params']})

                    fold_models.append(perform_run_train(data_train, data_test, model_params, times))

                    for fold_idx, (pcc, best) in enumerate(fold_models):
                        important_rules = extract_rules(pcc, best, k=3, min_n=10)
                        important_rules.to_csv(os.path.join(EXPORT_PATH, f"important_rules_{additional_params['dataset']}_fold_{fold_idx}.csv"))
                
            else:
                data_train, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()

                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')
                
                times['run_start_time'] = time.perf_counter()

                model_params.update({'encoding_params': data_config['encoding_params'],
                                     'transform_params': data_config['transform_params']})

                pcc, best = perform_run_train(data_train, data_test, model_params, times)

                important_rules = extract_rules(pcc, best, k=3, min_n=10)
                important_rules.to_csv(os.path.join(EXPORT_PATH, f"important_rules_{additional_params['dataset']}.csv"))

                    
def perform_run_train(data_train, data_test, model_params_train, times):
    
    pcc = ProcessContextClustering(model_params_train["encoding_params"],
                                   model_params_train["transform_params"],
                                   model_params_train["clustering_type"],
                                   model_params_train["cluster_params"])
    
    pcc.load_data(data_train, data_test)
    pad_cols = list(set(chain(*model_params_train["encoding_params"].values())).difference(set(['tsle', 'tsmn', 'tscs'])))
    pcc.prepare_train(pad_cols)
    pcc.prepare_test(pad_cols)
    
    times["cluster_start_time"] = time.perf_counter()
    pcc.fit()
    times["cluster_end_time"] = time.perf_counter()

    # get cluster memberships via predict (generates columns 'context_cluster' and 'activity_identifier_context' inside the train/test data frames)
    times["cluster_predict_start_time"] = time.perf_counter()
    pcc.predict()
    times["cluster_predict_end_time"] = time.perf_counter()

    best = BESTPredictor(max_pattern_size=model_params_train["max_pattern_size_train"],
                         process_stage_width_percentage=model_params_train["process_stage_width_percentage"],
                         min_freq=model_params_train["min_freq"],
                         prune_func=None)

    # transform ECDataset to SequenceData
    data_train_sd = SequenceData.from_ECDataset(pcc.data_train)
    data_test_sd = SequenceData.from_ECDataset(pcc.data_test)
    
    times['best_fitting_time_start'] = time.perf_counter()

    best.load_data(data_train_sd, data_test_sd)
    best.prepare_train(contextppm=True)
    best.fit()
    best.prepare_test(act_encoder=data_train_sd.act_encoder, 
                      filter_sequences=model_params_train['filter_sequences'], 
                      contextppm=True, attributes=data_train_sd.attribute_identifiers)

    times['best_fitting_time_end'] = time.perf_counter()

    return pcc, best

def extract_rules(pcc, best, k = 3, min_n = 10):

    # constructing data frame of all patterns (mining for each branch at first level of tree)
    # patterns all show the same center activity context cluster pair for one branch
    # can be used to infer about effects of certain center activity context cluster pairs w.r.t. outcomes (last context clusters)

    all_act_context_pairs = Counter([','.join([str(acp) for acp in p[0]]) for p in best._hca_patterns_by_size[1]])
    all_patterns_frame = pd.DataFrame()
    all_end_patterns_frame = pd.DataFrame()

    for acp, freq in all_act_context_pairs.items():

        if int(acp) not in [best.data_train.start_activity, 
                            best.data_train.end_activity] and acp in [p['name'] 
                                                                      for p in best._stage_trees[0]['children']]:

            acp_tree_idx = [p['name'] for p in best._stage_trees[0]['children']].index(acp)
            acp_tree = best._stage_trees[0]['children'][acp_tree_idx]

            # recursively mining all child patterns for that branch
            pattern_frame = _get_pattern_frame_from_tree_node(pd.DataFrame(), acp_tree)

            # disecting key characteristics of the patterns for easier filtering of the patterns
            pattern_frame['center_acp'] = acp
            remapped_acp = [k for k, v in best.data_train.act_mapping.items() if best.data_train.act_mapping[k] == int(acp)][0]
            center_activity, center_context_cluster = remapped_acp.split('_context_cluster_')
            pattern_frame['center_activity'] = center_activity
            pattern_frame['center_context_cluster'] = int(center_context_cluster)
            
            pattern_frame['pattern_size'] = pattern_frame['name'].apply(lambda x: len(x.split(',')))
            
            pattern_frame['first_end_token_idx'] = pattern_frame['name'].apply(lambda x: [int(acp) for acp 
                                                                                          in x.split(',')].index(best.data_train.end_activity) 
                                                                                          if best.data_train.end_activity in [int(acp) 
                                                                                                                              for acp in x.split(',')] else -1)
            pattern_frame['n_events_to_end'] = pattern_frame.apply(lambda x: int(x['first_end_token_idx'] - (x['pattern_size']-1)/2) 
                                                                   if x['first_end_token_idx'] != -1 else -1, axis=1)
            pattern_frame['last_start_token_idx'] = pattern_frame['name'].apply(lambda x: len(x.split(',')) - 1 - [int(acp) 
                                                                                                                   for acp in x.split(',')][::-1].index(best.data_train.start_activity) 
                                                                                                                   if best.data_train.start_activity in [int(acp) 
                                                                                                                                                         for acp in x.split(',')] else -1)
            pattern_frame['n_events_from_start'] = pattern_frame.apply(lambda x: int((x['pattern_size']-1)/2) - x['last_start_token_idx'] if x['last_start_token_idx'] != -1 else -1, axis=1)
            
            pattern_frame['next_acp'] = pattern_frame[['name', 'pattern_size']].apply(lambda x: int(x['name'].split(',')[int((x['pattern_size'] - 1) / 2) + 1]) 
                                                                                      if len(x['name'].split(',')) > 1 else -1, axis=1)
            pattern_frame['second_acp'] = pattern_frame['name'].apply(lambda x: int(x.split(',')[1]) if len(x.split(',')) > 1 else -1)
            pattern_frame['third_acp'] = pattern_frame['name'].apply(lambda x: int(x.split(',')[2]) if len(x.split(',')) > 1 else -1)
            
            # last acp before pattern shows the end token - this contains the last seen context cluster of the pattern
            # we use this last context cluster for our inference later on
            pattern_frame['last_acp'] = pattern_frame[['name', 'pattern_size', 'n_events_to_end']].apply(lambda x: int(x['name'].split(',')[int((x['pattern_size'] - 1) / 2) + x['n_events_to_end'] - 1]) if len(x['name'].split(',')) > 1 and x['n_events_to_end'] != -1 else -1, axis=1)

            # recording the last acp inside the pattern (not the last acp before we see an end token (see above))
            pattern_frame['pattern_last_acp'] = pattern_frame['name'].apply(lambda x: int(x.split(',')[-1]))

            # filtering single acp patterns and patterns where we see end tokens as next activity context cluster pair
            pattern_frame = pattern_frame[pattern_frame['pattern_size'] > 1]
            pattern_frame = pattern_frame[pattern_frame['next_acp'] != best.data_train.end_activity]

            if len(pattern_frame) > 0:

                # pulling last activity and last context cluster from the last acp value for easier readability of our identified rules
                remapped_last_activity_context_pairs = [[k for k, v in best.data_train.act_mapping.items() 
                                                         if best.data_train.act_mapping[k] == acp][0] 
                                                         if acp != -1 else -1 for acp in pattern_frame['last_acp']]
                last_activities, last_context_clusters = zip(*[acp.split('_context_cluster_') 
                                                               if acp != -1 else (-1, -1) for acp in remapped_last_activity_context_pairs])
                pattern_frame['last_activity'] = last_activities
                pattern_frame['last_context_cluster'] = [int(cc) for cc in last_context_clusters]

                # filling end_pattern_frame with patterns that show 
                end_pattern_frame = pattern_frame[pattern_frame['pattern_last_acp']==best.data_train.end_activity]
                n_events_to_end = ((end_pattern_frame['pattern_size'] - 1) / 2).astype(int)
                end_pattern_frame = end_pattern_frame.assign(n_events_to_end_end_frame=n_events_to_end)

                # concatenating branch pattern frames to global pattern frames for general analysis of relationships uncoverable from the tree
                all_patterns_frame = pd.concat([all_patterns_frame, pattern_frame])
                all_end_patterns_frame = pd.concat([all_end_patterns_frame, end_pattern_frame])

    # looking at decoded cluster representatives
    decoding = Decoding(pcc.data_train.encoders)
    cluster_means_decoded = list()

    if pcc.clustering_type == 'KMeans':
        for cluster_idx in range(0, len(pcc.event_clustering.cluster_centers_)):
            cluster_means_decoded.append(decoding.decode_samples([pcc.event_clustering.cluster_centers_[cluster_idx].reshape(1, -1)]))
    elif pcc.clustering_type == 'GaussianMixture':
        for cluster_idx in range(0, len(pcc.event_clustering.means_)):
            cluster_means_decoded.append(decoding.decode_samples([pcc.event_clustering.means_[cluster_idx].reshape(1, -1)]))

    cluster_means_dict = {cluster_idx: round(mean, ndigits=2) 
                          for cluster_idx, mean in zip(range(len(cluster_means_decoded)), [float([_ for _ in v.values()][0][0])/60/60/24 for v in cluster_means_decoded])}
    cluster_means_dict_sorted = {k: v for k, v in sorted(cluster_means_dict.items(), key=lambda x: x[1])}
    cluster_remap_dict = {c_orig: c_remap for c_orig, c_remap in zip(cluster_means_dict_sorted.keys(), cluster_means_dict.keys())}
    cluster_inverse_remap_dict = {c_remap: c_orig for c_orig, c_remap 
                                  in zip(cluster_means_dict_sorted.keys(), cluster_means_dict.keys())}
    cluster_remap_dict[-1] = -1
    cluster_inverse_remap_dict[-1] = -1

    all_patterns_frame['last_context_cluster'] = all_patterns_frame['last_context_cluster'].map(cluster_remap_dict)
    all_patterns_frame['center_context_cluster'] = all_patterns_frame['center_context_cluster'].map(cluster_remap_dict)
    
    query_frame = all_patterns_frame[(all_patterns_frame['pattern_size']==21) & 
                                     (all_patterns_frame['n_events_to_end']!=-1) & 
                                     (all_patterns_frame['n_events_from_start']!=-1)]

    # pull frequency information into the frame by multiplying each row with its freq value to honor pattern frequencies
    query_frame = query_frame.reset_index(drop=True)
    query_frame = query_frame.loc[query_frame.index.repeat(query_frame['freq'])].reset_index(drop=True)

    # putting n_obs into the query frame
    query_frame = query_frame.assign(n = query_frame.groupby(['last_context_cluster', 'center_acp']).transform('size'),
                                        n_last_context_cluster  = query_frame.groupby('last_context_cluster').transform('size',),
                                        n_center_acp  = query_frame.groupby('center_acp').transform('size',)
                                        )

    grouped_stats = query_frame.groupby(['last_context_cluster', 
                                            'center_acp', 
                                            'center_activity', 
                                            'center_context_cluster']).apply(lambda x: pd.Series({'mean_start': x['n_events_from_start'].mean(),
                                                                                                'median_start': x['n_events_from_start'].median(),
                                                                                                'sd_start': x['n_events_from_start'].std(),
                                                                                                'mean_end': x['n_events_to_end'].mean(),
                                                                                                'median_end': x['n_events_to_end'].median(),
                                                                                                'sd_end': x['n_events_to_end'].std(),
                                                                                                'n': x['n'].max(),
                                                                                                'n_last_context_cluster': x['n_last_context_cluster'].max(),
                                                                                                'n_center_acp': x['n_center_acp'].max(),
                                                                                                }), include_groups=False).reset_index(drop=False)

    grouped_stats['support'] = grouped_stats['n'] / len(query_frame)
    grouped_stats['P_center_acp'] = grouped_stats['n_center_acp'] / len(query_frame)
    grouped_stats['P_last_context_cluster'] = grouped_stats['n_last_context_cluster'] / len(query_frame)
    grouped_stats['P_center_acp_cond_last_context'] = grouped_stats['n'] / grouped_stats['n_last_context_cluster'] # P(center_acp | last_context_cluster)
    grouped_stats['P_last_context_cond_center_acp'] = grouped_stats['n'] / grouped_stats['n_center_acp'] # P(last_context_cluster | center_acp)
    grouped_stats['lift_last_context_cluster'] = grouped_stats['P_last_context_cond_center_acp'] / grouped_stats['P_last_context_cluster'] # P(last_context_cluster | center_acp) / P(last_context_cluster)
    
    top_k_lift_rows = grouped_stats[grouped_stats['n'] >= min_n].groupby('last_context_cluster').apply(lambda x: x.nlargest(k, 'lift_last_context_cluster'), include_groups=False).reset_index(drop=False)
    
    important_rules = top_k_lift_rows[['center_activity', 'center_context_cluster', 'last_context_cluster', 
                                       'mean_start', 'mean_end', 'n', 'support', 'P_last_context_cond_center_acp', 
                                       'lift_last_context_cluster']][top_k_lift_rows['lift_last_context_cluster']>1].sort_values(by=['last_context_cluster', 
                                                                                                                                     'lift_last_context_cluster'], 
                                                                                                                                 ascending=[True, False])
    important_rules['last_cluster_avg_case_duration'] = important_rules['last_context_cluster'].apply(lambda x: cluster_means_dict[cluster_inverse_remap_dict[x]])

    return important_rules

def _get_pattern_frame_from_tree_node(df: pd.DataFrame, tree_node: dict, keys_to_add: list[str] = ['name', 'prob', 'global_prob', 'freq', 'rpif_dist', 'log_rpif_dist']) -> pd.DataFrame:

    node_df = pd.DataFrame([{k: tree_node[k] for k in keys_to_add if k in tree_node}])
    df = pd.concat([df, node_df])
    children = tree_node['children']

    for child in children:
        df = _get_pattern_frame_from_tree_node(df, child)
    
    return df


if __name__=='__main__':
    main()