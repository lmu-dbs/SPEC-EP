import time
import os
import numpy as np
import random
from itertools import chain

from best4ppm.data.sequencedata import SequenceData
from best4ppm.util.model_logging import log_to_csv
from best4ppm.eval.evaluator import NAPEvaluator, SFXEvaluator
from best4ppm.models.best import BESTPredictor

from specep.dataset.ecdataset import ECDataset
from specep.util.config_utils import read_config
from specep.eval.evaluator import NextContextEvaluator, SFXContextEvaluator
from specep.clustering.pcc import ProcessContextClustering 
from specep.clustering.util import sample_from_component
from specep.encoding.util import Decoding
from specep.util.parallelization import warmup_worker_pool

from specep.util.logging import init_logging
logger = init_logging(__name__, 'main.log')

from util.paths import CONFIG_PATH, DATA_PATH, EXPORT_PATH
from util.combinations import param_combinations

def main():

    general_config = read_config(os.path.join(CONFIG_PATH, "general_config.yml"))
    data_configs = read_config(os.path.join(CONFIG_PATH, "data_configs.yml"))
    model_configs = read_config(os.path.join(CONFIG_PATH, "model_configs.yml"))

    FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')

    os.makedirs(FIG_EXP_DIR, exist_ok=True)
    
    if general_config['parallelization_lib']=='joblib':
        warmup_worker_pool(ncores=general_config['ncores'])

    for dataset in general_config["dataset"]:

        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f"desired datset {dataset} not found in data_config.yml",)
            raise

        all_attributes = list()
        for v in data_config['encoding_params'].values():
            all_attributes.extend(v)

        all_attributes = list(set(all_attributes))

        model_config = model_configs[general_config['model_config']]
        model_config_train = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_eval'}
        model_config_eval = {key: model_config[key] for key in model_config.keys() if key!='max_pattern_size_train'}
        combinations_generator = param_combinations(model_config)
        config_combinations = [c_comb for c_comb in combinations_generator]
        additional_params = dict()
        
        cv_hashes = [random.getrandbits(128) for _ in range(0, len(config_combinations))]
        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset
        additional_params['parallelization_lib'] = general_config['parallelization_lib']

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
                
                all_fold_times = {fold_idx: dict() for fold_idx in range(len(folds))}

                for fold_idx, fold in enumerate(folds):

                    data_train, data_test = fold
                    fold_times = all_fold_times[fold_idx]
                    fold_times['run_start_time'] = time.perf_counter()

                    model_params.update({'encoding_params': data_config['encoding_params'],
                                         'transform_params': data_config['transform_params']})

                    fold_models.append(perform_run_train(data_train, data_test, model_params, fold_times, parallelization_lib=general_config['parallelization_lib']))
                
                for time_key in all_fold_times[0].keys():
                    times[time_key] = [all_fold_times[fold_idx][time_key] for fold_idx in range(len(folds))]
                
                for eval_pattern_size_idx, eps in enumerate(model_params['max_pattern_size_eval']):
                        
                    for fold_idx, (pcc, best) in enumerate(fold_models):
                                                        
                        model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                        model_params_eval['max_pattern_size_eval'] = eps

                        run_log_params_metrics = {}

                        run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                        run_log_params_metrics['n_process_stages'] = len(best._stages)

                        model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                        model_params_eval['max_pattern_size_eval'] = eps
                        run_log_params_metrics['random_seed'] = additional_params['seed']
                        run_log_params_metrics['parallelization_lib'] = additional_params['parallelization_lib']

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
                        
                        perform_run_test(best, pcc, model_params_eval, general_config, times, run_log_params_metrics, fold_idx, all_attributes=all_attributes)
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

                pcc, best = perform_run_train(data_train, data_test, model_params, times, parallelization_lib=general_config['parallelization_lib'])

                for eps in model_params['max_pattern_size_eval']:
                    
                    model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                    model_params_eval['max_pattern_size_eval'] = eps

                    run_log_params_metrics = {}
                    run_log_params_metrics['process_stage_width'] = best._abs_process_stage_width
                    run_log_params_metrics['n_process_stages'] = len(best._stages)
                    
                    model_params_eval = {key: model_params[key] for key in model_params.keys() if key!='max_pattern_size_eval'}
                    model_params_eval['max_pattern_size_eval'] = eps

                    run_log_params_metrics['random_seed'] = additional_params['seed']
                    run_log_params_metrics['parallelization_lib'] = additional_params['parallelization_lib']

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
                        
                    run_log_params_metrics['base_cv_hash'] = 'single_fold_run'
                    run_log_params_metrics['cv_hash'] = 'single_fold_run'

                    perform_run_test(best, pcc, model_params_eval, general_config, times, run_log_params_metrics, all_attributes=all_attributes)
                    log_to_csv(csv_file=os.path.join(EXPORT_PATH, 'model_params_metrics.csv'), params_metrics=run_log_params_metrics)

def perform_run_train(data_train, data_test, model_params_train, times, parallelization_lib):
    
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
                         prune_func=None,
                         parallelization_lib=parallelization_lib)

    # transform ECDataset to SequenceData
    data_train_sd = SequenceData.from_ECDataset(pcc.data_train)
    data_test_sd = SequenceData.from_ECDataset(pcc.data_test)
    
    times['best_fitting_time_start'] = time.perf_counter()

    best.load_data(data_train_sd, data_test_sd)
    best.prepare_train(specep=True)
    best.fit()
    best.prepare_test(act_encoder=data_train_sd.act_encoder, filter_sequences=model_params_train['filter_sequences'], specep=True, attributes=data_train_sd.attribute_identifiers)

    times['best_fitting_time_end'] = time.perf_counter()

    return pcc, best

def perform_run_test(pred_model: BESTPredictor, cluster_model: ProcessContextClustering, model_params_eval: dict, general_config: dict, times: dict, param_metric_dict: dict, fold_idx: int = None, all_attributes: list[str] = list()):
    
    times['prediction_start_time_nep'] = time.perf_counter()
    times['prediction_start_time_sfx'] = time.perf_counter()
    decoding = Decoding(cluster_model.data_train.encoders)

    if 'nep' in model_params_eval['task']:
        nep_predictions, nep_pred_dur, nep_pred_convert_dur = pred_model.predict(task='nep', eval_pattern_size=model_params_eval['max_pattern_size_eval'],
                                                                                 break_buffer=model_params_eval['break_buffer'], 
                                                                                 filter_tokens=model_params_eval['filter_sequences'], 
                                                                                 ncores=general_config['ncores'])
        times['nep_pred_duration'] = nep_pred_dur
        times['nep_pred_convert_duration'] = nep_pred_convert_dur
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
            if 'tsle' in all_attributes:
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
                            param_metric_dict[f"next_{attribute}_mae{perfect}"] = float("nan")
                            param_metric_dict[f"next_{attribute}_rmse{perfect}"] = float("nan")
                        else:
                            mae = next_retransformed_samples_eval.calc_mae()
                            rmse = next_retransformed_samples_eval.calc_rmse()
                            if attribute in ['tsle', 'tsmn', 'tscs']:
                                mae_days = mae/60/60/24
                                rmse_days = rmse/60/60/24
                            logger.info(f"Next Attribute MAE{perfect_verbose} - {attribute}: {mae:.4f}{' - ' + str(round(mae_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                            logger.info(f"Next Attribute RMSE{perfect_verbose} - {attribute}: {rmse:.4f}{' - ' + str(round(rmse_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                            param_metric_dict[f"next_{attribute}_accuracy{perfect}"] = float("nan")
                            param_metric_dict[f"next_{attribute}_balanced_accuracy{perfect}"] = float("nan")
                            param_metric_dict[f"next_{attribute}_mae{perfect}"] = mae
                            param_metric_dict[f"next_{attribute}_rmse{perfect}"] = rmse
            else:
                for attribute in ['tsle']:
                    param_metric_dict[f"next_{attribute}_accuracy{perfect}"] = float("nan")
                    param_metric_dict[f"next_{attribute}_balanced_accuracy{perfect}"] = float("nan")
                    param_metric_dict[f"next_{attribute}_mae{perfect}"] = float("nan")
                    param_metric_dict[f"next_{attribute}_rmse{perfect}"] = float("nan")

        times['nep_eval_time'] = time.perf_counter()
        times['prediction_start_time_sfx'] = time.perf_counter()
    
    if 'sfx' in model_params_eval['task']:
        sfx_predictions, sfx_pred_dur, sfx_pred_convert_dur = pred_model.predict(task='sfx', 
                                                                                 eval_pattern_size=model_params_eval['max_pattern_size_eval'], 
                                                                                 break_buffer=model_params_eval['break_buffer'], 
                                                                                 filter_tokens=model_params_eval['filter_sequences'], 
                                                                                 ncores=general_config['ncores'])
        times['sfx_pred_duration'] = sfx_pred_dur
        times['sfx_pred_convert_duration'] = sfx_pred_convert_dur
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
            if 'tscs' in all_attributes:
                for attribute in ['tscs']:
                    if attribute in sfx_retransformed_samples.keys():
                        sfx_retransformed_samples_eval = SFXContextEvaluator(pred=sfx_retransformed_samples[attribute], actual=pred_model.data_test.full_future_attribute_sequences[attribute], attribute=attribute)
                        if isinstance(sfx_retransformed_samples[attribute][0][0], str):
                            att_ndls = sfx_retransformed_samples_eval.calc_ndls()
                            logger.info(f"Attribute NDLS{perfect_verbose} - {attribute}: {att_ndls:.4f}")
                            param_metric_dict[f"{attribute}_sfx_similarity{perfect}"] = att_ndls
                            param_metric_dict[f"last_{attribute}_mae{perfect}"] = float("nan")
                            param_metric_dict[f"last_{attribute}_rmse{perfect}"] = float("nan")
                        else:
                            mae_last = sfx_retransformed_samples_eval.calc_mae_last()
                            rmse_last = sfx_retransformed_samples_eval.calc_rmse_last()
                            if attribute in ['tsle', 'tsmn', 'tscs']:
                                mae_last_days = mae_last/60/60/24
                                rmse_last_days = rmse_last/60/60/24
                            logger.info(f"Last Attribute MAE{perfect_verbose} - {attribute}: {mae_last:.4f}{' - ' + str(round(mae_last_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                            logger.info(f"Last Attribute RMSE{perfect_verbose} - {attribute}: {rmse_last:.4f}{' - ' + str(round(rmse_last_days, ndigits=4)) + ' days' if attribute in ['tsle', 'tsmn', 'tscs'] else ''}")
                            param_metric_dict[f"{attribute}_sfx_similarity{perfect}"] = float("nan")
                            param_metric_dict[f"last_{attribute}_mae{perfect}"] = mae_last
                            param_metric_dict[f"last_{attribute}_rmse{perfect}"] = rmse_last
            else:
                for attribute in ['tscs']:
                    param_metric_dict[f"{attribute}_sfx_similarity{perfect}"] = float("nan")
                    param_metric_dict[f"last_{attribute}_mae{perfect}"] = float("nan")
                    param_metric_dict[f"last_{attribute}_rmse{perfect}"] = float("nan")
            
            if 'tsle' in all_attributes:
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
            else:
                for attribute in ['tsle']:
                    param_metric_dict[f"cumsum_{attribute}_mae{perfect}"] = float("nan")
                    param_metric_dict[f"cumsum_{attribute}_rmse{perfect}"] = float("nan")
        
        times['sfx_eval_time'] = time.perf_counter()

    times['run_end_time'] = time.perf_counter()

    calc_times = calc_runtimes(recorded_times=times, fold_idx=fold_idx)
    
    for key, value in calc_times.items():
        param_metric_dict[key] = value

def calc_runtimes(recorded_times: dict, fold_idx: int = None):
    
    calculated_runtimes = dict()
    calculated_runtimes['prep_duration'] = recorded_times['data_prep_time'] - recorded_times['start_time']
    
    if fold_idx is not None: # we have multiple fold runs
        calculated_runtimes['cluster_fit_duration'] = recorded_times['cluster_end_time'][fold_idx] - recorded_times['cluster_start_time'][fold_idx]
        calculated_runtimes['cluster_pred_duration'] = recorded_times['cluster_predict_end_time'][fold_idx] - recorded_times['cluster_predict_start_time'][fold_idx]
        calculated_runtimes['total_fit_duration'] = recorded_times['best_fitting_time_end'][-1] - recorded_times['run_start_time'][0]
        calculated_runtimes['best_fit_duration'] = recorded_times['best_fitting_time_end'][fold_idx] - recorded_times['best_fitting_time_start'][fold_idx]
        calculated_runtimes['fit_duration_per_fold'] = recorded_times['best_fitting_time_end'][fold_idx] - recorded_times['run_start_time'][fold_idx]
        
    else: # we have a single fold run    
        calculated_runtimes['cluster_fit_duration'] = recorded_times['cluster_end_time'] - recorded_times['cluster_start_time']
        calculated_runtimes['cluster_pred_duration'] = recorded_times['cluster_predict_end_time'] - recorded_times['cluster_predict_start_time']
        calculated_runtimes['best_fit_duration'] = recorded_times['best_fitting_time_end'] - recorded_times['best_fitting_time_start']
        calculated_runtimes['total_fit_duration'] = recorded_times['best_fitting_time_end'] - recorded_times['run_start_time']
        calculated_runtimes['fit_duration_per_fold'] = calculated_runtimes['total_fit_duration']
    
    calculated_runtimes['nep_duration'] = recorded_times['nep_finish_time'] - recorded_times['prediction_start_time_nep']
    calculated_runtimes['nep_eval_duration'] = recorded_times['nep_eval_time'] - recorded_times['nep_finish_time']
    calculated_runtimes['sfx_duration'] = recorded_times['sfx_finish_time'] - recorded_times['prediction_start_time_sfx']
    calculated_runtimes['sfx_eval_duration'] = recorded_times['sfx_eval_time'] - recorded_times['sfx_finish_time']
    
    if fold_idx is not None:
        calculated_runtimes['total_run_time'] = recorded_times['run_end_time'] - recorded_times['run_start_time'][0]
    else:
        calculated_runtimes['total_run_time'] = recorded_times['run_end_time'] - recorded_times['run_start_time']
    
    return calculated_runtimes
    

if __name__=='__main__':
    main()