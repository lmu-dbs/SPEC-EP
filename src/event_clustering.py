import time
import os
import numpy as np
import random
from itertools import chain

from best4ppm.data.sequencedata import SequenceData
from best4ppm.util.model_logging import log_to_csv
from best4ppm.eval.evaluator import NAPEvaluator, SFXEvaluator
from best4ppm.models.best import BESTPredictor

from contextppm.dataset.ECDataset import ECDataset
from contextppm.util.config_utils import read_config
from contextppm.eval.evaluator import NextContextEvaluator, SFXContextEvaluator
from contextppm.clustering.ProcessContextClustering import ProcessContextClustering 
from contextppm.clustering.util import sample_from_component
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

                # model_params = dict(zip(list(model_config.keys()), [param for param in combination]))
                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                fold_models = list()

                for fold in folds:

                    data_train, data_test = fold
                    times['run_start_time'] = time.perf_counter()

                    model_params.update({'encoding_params': data_config['encoding_params'],
                                         'transform_params': data_config['transform_params']})

                    fold_models.append(perform_run_train(data_train, data_test, model_params, times))
                
                for eval_pattern_size_idx, eps in enumerate(model_params['max_pattern_size_eval']):
                        
                    for pcc, best in fold_models:
                                                        
                        model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                        model_params_eval['max_pattern_size_eval'] = eps

                        run_log_params_metrics = {}

                        run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                        run_log_params_metrics['n_process_stages'] = len(best._stages)

                        model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                        model_params_eval['max_pattern_size_eval'] = eps
                        run_log_params_metrics['random_seed'] = additional_params['seed']

                        model_and_general_params = {**model_params,
                                                    'max_pattern_size_eval': eps,
                                                    'n_clusters': model_params['cluster_params']['n_clusters'],
                                                    **{'model_config':general_config['model_config'],
                                                        'ncores':general_config['ncores'],
                                                        'cv_folds':general_config.get('cv_folds'),
                                                        'train_pct':general_config.get('train_pct')},
                                                        'dataset':additional_params['dataset']}

                        for key, value in model_and_general_params.items():
                            run_log_params_metrics[key] = value

                        run_log_params_metrics['base_cv_hash'] = base_cv_hash
                        run_log_params_metrics['cv_hash'] = f'{base_cv_hash}_{eval_pattern_size_idx}'

                        perform_run_test(best, pcc, model_params_eval, general_config, times, run_log_params_metrics)
                        log_to_csv(csv_file=os.path.join(EXPORT_PATH, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

            else:
                data_train, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()
                
                # model_params = dict(zip(list(model_config.keys()), [param for param in combination]))

                if model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')
                
                times['run_start_time'] = time.perf_counter()

                model_params.update({'encoding_params': data_config['encoding_params'],
                                     'transform_params': data_config['transform_params']})

                pcc, best = perform_run_train(data_train, data_test, model_params, times)

                for eps in model_params['max_pattern_size_eval']:
                    
                    model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                    model_params_eval['max_pattern_size_eval'] = eps

                    run_log_params_metrics = {}
                    run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                    run_log_params_metrics['n_process_stages'] = len(best._stages)
                    
                    model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                    model_params_eval['max_pattern_size_eval'] = eps

                    run_log_params_metrics['random_seed'] = additional_params['seed']

                    model_and_general_params = {**model_params,
                                                'max_pattern_size_eval': eps,
                                                'n_clusters': model_params['cluster_params']['n_clusters'],
                                                **{'model_config':general_config['model_config'],
                                                'ncores':general_config['ncores'],
                                                'cv_folds':general_config.get('cv_folds'),
                                                'train_pct':general_config.get('train_pct')},
                                                'dataset':additional_params['dataset']}

                    for key, value in model_and_general_params.items():
                        run_log_params_metrics[key] = value

                    perform_run_test(best, pcc, model_params_eval, general_config, times, run_log_params_metrics)
                    log_to_csv(csv_file=os.path.join(EXPORT_PATH, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

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
    best.prepare_test(act_encoder=data_train_sd.act_encoder, filter_sequences=model_params_train['filter_sequences'], contextppm=True, attributes=data_train_sd.attribute_identifiers)

    times['best_fitting_time_end'] = time.perf_counter()

    return pcc, best

def perform_run_test(pred_model: BESTPredictor, cluster_model: ProcessContextClustering, model_params_eval, general_config, times, param_metric_dict):
    
    times['prediction_start_time_nep'] = time.perf_counter()
    times['prediction_start_time_sfx'] = time.perf_counter()
    decoding = Decoding(cluster_model.data_train.encoders)

    if 'nep' in model_params_eval['task']:
        nep_predictions = pred_model.predict(task='nep', eval_pattern_size=model_params_eval['max_pattern_size_eval'],
                                             break_buffer=model_params_eval['break_buffer'], 
                                             filter_tokens=model_params_eval['filter_sequences'], 
                                             ncores=general_config['ncores'])
        times['nep_finish_time'] = time.perf_counter()
        
        # evaluation

        # NAP evaluation
        nap_eval = NAPEvaluator(pred=nep_predictions, actual=pred_model.data_test.next_activities, split_context=True, act_encoder=pred_model.data_train.act_encoder)
        none_share = nap_eval.get_nan_share()
        nap_acc = nap_eval.calc_accuracy_score()
        nap_balanced_acc = nap_eval.calc_balanced_accuracy_score()
        logger.info(f'None share of predictions: {none_share:.4f}')
        logger.info(f'NAP accuracy: {nap_acc:.4f}')
        logger.info(f'NAP balanced accuracy: {nap_balanced_acc:.4f}')

        param_metric_dict['none_share'] = none_share
        param_metric_dict['nap_accuracy'] = nap_acc
        param_metric_dict['nap_balanced_accuracy'] = nap_balanced_acc
        
        for perfect_cluster_forecast in [True, False]:
            if perfect_cluster_forecast:
                next_context_predictions = nap_eval.actual_context
                perfect = "_perfect"
                perfect_verbose = " (perfect cluster information)"
            else:
                next_context_predictions = nap_eval.pred_context
                perfect = ""
                perfect_verbose = ""

            next_context_actuals = nap_eval.actual_context
            
            # next cluster accuracy
            next_context_eval = NextContextEvaluator(pred=next_context_predictions, actual=next_context_actuals)
            next_context_acc = next_context_eval.calc_accuracy_score()
            next_context_balanced_acc = next_context_eval.calc_balanced_accuracy_score()
            logger.info(f"Next context accuracy{perfect_verbose}: {next_context_acc:.4f}")
            logger.info(f"Next context balanced accuracy{perfect_verbose}: {next_context_balanced_acc:.4f}")

            param_metric_dict[f"ncp_accuracy{perfect}"] = next_context_acc
            param_metric_dict[f"ncp_balanced_accuracy{perfect}"] = next_context_balanced_acc

            # sample from clusters with cluster model
            next_cluster_samples = [sample_from_component(cluster_model.event_clustering, predicted_component, 1) 
                                    for predicted_component in next_context_predictions]

            next_retransformed_samples = decoding.decode_samples(next_cluster_samples)

            # Next Timestamp Prediction (NTP) evaluation
            for attribute in ['tsle']:
                if attribute in next_retransformed_samples.keys():
                    next_retransformed_samples_eval = NextContextEvaluator(pred=next_retransformed_samples[attribute], actual=pred_model.data_test.next_attributes[attribute], attribute=attribute)
                    if isinstance(next_retransformed_samples[attribute][0], str):
                        att_acc = next_retransformed_samples_eval.calc_accuracy_score()
                        att_balanced_acc = next_retransformed_samples_eval.calc_balanced_accuracy_score()
                        logger.info(f"Next Attribute accuracy{perfect_verbose} - {attribute}: {att_acc:.4f}")
                        logger.info(f"Next Attribute balanced accuracy{perfect_verbose} - {attribute}: {att_balanced_acc:.4f}")

                        param_metric_dict[f"next_{attribute}_accuracy{perfect}"] = att_acc
                        param_metric_dict[f"next_{attribute}_balanced_accuracy{perfect}"] = att_balanced_acc
                    else:
                        mae = next_retransformed_samples_eval.calc_mae()
                        rmse = next_retransformed_samples_eval.calc_rmse()
                        if attribute in ['tsle', 'tsmn', 'tscs']:
                            mae_days = mae/60/60/24
                            rmse_days = rmse/60/60/24
                        logger.info(f"Next Attribute MAE{perfect_verbose} - {attribute}: {mae:.4f}{' - ' + str(round(mae_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                        logger.info(f"Next Attribute RMSE{perfect_verbose} - {attribute}: {rmse:.4f}{' - ' + str(round(rmse_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                        param_metric_dict[f"next_{attribute}_mae{perfect}"] = mae
                        param_metric_dict[f"next_{attribute}_rmse{perfect}"] = rmse

        times['nep_eval_time'] = time.perf_counter()
        times['prediction_start_time_sfx'] = time.perf_counter()
    
    if 'sfx' in model_params_eval['task']:
        sfx_predictions = pred_model.predict(task='sfx', 
                                             eval_pattern_size=model_params_eval['max_pattern_size_eval'], 
                                             break_buffer=model_params_eval['break_buffer'], 
                                             filter_tokens=model_params_eval['filter_sequences'], 
                                             ncores=general_config['ncores'])
        times['sfx_finish_time'] = time.perf_counter()

        # evaluation

        # activity suffix evaluation
        sfx_eval = SFXEvaluator(pred=sfx_predictions, actual=pred_model.data_test.full_future_sequences, split_context=True, act_encoder=pred_model.data_train.act_encoder)
        ndls = sfx_eval.calc_ndls(ncores=general_config['ncores'])
        logger.info(f"SFX similarity: {ndls:.4f}")
        param_metric_dict["sfx_similarity"] = ndls
        
        for perfect_cluster_forecast in [True, False]:
            if perfect_cluster_forecast:
                sfx_context_predictions = sfx_eval.actual_context
                perfect = "_perfect"
                perfect_verbose = " (perfect cluster information)"
            else:
                sfx_context_predictions = sfx_eval.pred_context
                perfect = ""
                perfect_verbose = ""

            sfx_context_actuals = sfx_eval.actual_context

            # suffix cluster NDLS
            sfx_context_eval = SFXContextEvaluator(pred=sfx_context_predictions, actual=sfx_context_actuals)
            context_ndls = sfx_context_eval.calc_ndls(ncores=general_config['ncores'])
            logger.info(f"SFX context similarity{perfect_verbose}: {context_ndls:.4f}")
            param_metric_dict[f"sfx_context_similarity{perfect}"] = context_ndls
            
            sfx_cluster_samples = [[sample_from_component(cluster_model.event_clustering, predicted_component, 1) 
                                    for predicted_component in pred_context_suffix] for pred_context_suffix in sfx_context_predictions]
            sfx_retransformed_samples = decoding.decode_sample_sequences(sfx_cluster_samples)

            # Remaining Time Prediction (RTP) evaluation
            for attribute in ['tscs']:
                if attribute in sfx_retransformed_samples.keys():
                    sfx_retransformed_samples_eval = SFXContextEvaluator(pred=sfx_retransformed_samples[attribute], actual=pred_model.data_test.full_future_attribute_sequences[attribute], attribute=attribute)
                    if isinstance(sfx_retransformed_samples[attribute][0][0], str):
                        att_ndls = sfx_retransformed_samples_eval.calc_ndls()
                        logger.info(f"Attribute NDLS{perfect_verbose} - {attribute}: {att_ndls:.4f}")
                        param_metric_dict[f"{attribute}_sfx_similarity{perfect}"] = att_ndls
                    else:
                        mae_last = sfx_retransformed_samples_eval.calc_mae_last()
                        rmse_last = sfx_retransformed_samples_eval.calc_rmse_last()
                        if attribute in ['tsle', 'tsmn', 'tscs']:
                            mae_last_days = mae_last/60/60/24
                            rmse_last_days = rmse_last/60/60/24
                        logger.info(f"Last Attribute MAE{perfect_verbose} - {attribute}: {mae_last:.4f}{' - ' + str(round(mae_last_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                        logger.info(f"Last Attribute RMSE{perfect_verbose} - {attribute}: {rmse_last:.4f}{' - ' + str(round(rmse_last_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                        param_metric_dict[f"last_{attribute}_mae{perfect}"] = mae_last
                        param_metric_dict[f"last_{attribute}_rmse{perfect}"] = rmse_last
            
            # cumulative sum of predicted attribute suffix values evaluation
            for attribute in ['tsle']:
                if attribute in sfx_retransformed_samples.keys():
                    sfx_retransformed_samples_eval = SFXContextEvaluator(pred=sfx_retransformed_samples[attribute], actual=pred_model.data_test.full_future_attribute_sequences[attribute], attribute=attribute)
                    if isinstance(sfx_retransformed_samples[attribute][0][0], str):
                        logger.warning(f"Cannot evaluate cumulative sum of non-numeric attribute suffix predictions - {attribute}")
                    else:
                        mae_cumsum = sfx_retransformed_samples_eval.calc_mae_cumsum(truncate_negative=True)
                        rmse_cumsum = sfx_retransformed_samples_eval.calc_rmse_cumsum(truncate_negative=True)
                        mae_cumsum_days = mae_cumsum/60/60/24
                        rmse_cumsum_days = rmse_cumsum/60/60/24
                        logger.info(f"Cumsum Attribute MAE{perfect_verbose} - {attribute}: {mae_cumsum:.4f}{' - ' + str(round(mae_cumsum_days, ndigits=4)) + ' days'}")
                        logger.info(f"Cumsum Attribute RMSE{perfect_verbose} - {attribute}: {rmse_cumsum:.4f}{' - ' + str(round(rmse_cumsum_days, ndigits=4)) + ' days'}")
                        param_metric_dict[f"cumsum_{attribute}_mae{perfect}"] = mae_cumsum
                        param_metric_dict[f"cumsum_{attribute}_rmse{perfect}"] = rmse_cumsum
        
        times['sfx_eval_time'] = time.perf_counter()

    times['run_end_time'] = time.perf_counter()

    calc_times = dict()
    calc_times['prep_duration'] = times['data_prep_time'] - times['start_time']
    calc_times['cluster_fit_duration'] = times['cluster_end_time'] - times['cluster_end_time']
    calc_times['cluster_pred_duration'] = times['cluster_predict_end_time'] - times['cluster_predict_end_time']
    calc_times['best_fit_duration'] = times['best_fitting_time_end'] - times['best_fitting_time_start']
    calc_times['total_fit_duration'] = times['best_fitting_time_end'] - times['run_start_time']
    calc_times['fit_duration_per_fold'] = calc_times['total_fit_duration'] / general_config['cv_folds']
    calc_times['nep_duration'] = times['nep_finish_time'] - times['prediction_start_time_nep']
    calc_times['nep_eval_duration'] = times['nep_eval_time'] - times['nep_finish_time']
    calc_times['sfx_duration'] = times['sfx_finish_time'] - times['prediction_start_time_sfx']
    calc_times['sfx_eval_duration'] = times['sfx_eval_time'] - times['sfx_finish_time']
    calc_times['total_run_time'] = times['run_end_time'] - times['run_start_time']
    for key, value in calc_times.items():
        param_metric_dict[key] = value

if __name__=='__main__':
    main()