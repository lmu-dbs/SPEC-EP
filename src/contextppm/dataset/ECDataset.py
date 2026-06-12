from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import os
import math
import datetime as dt
import itertools
from ..util.sequence_utils import _filter_start_end
from ..util.logging import init_logging
from ..encoding.util import EncodingFactory, TransformFactory

from sklearn.preprocessing import LabelEncoder

logger = init_logging(__name__, "ECDataset.log")


class ECDataset:
    """Data class for preprocessing of .csv files into sequences that can be used for subtrace clustering"""

    def __init__(
        self, data, case_identifier, activity_identifier, timestamp_identifier, **kwargs
    ):
        self.logger = logger

        self.logger.info("Initializing ECDataset...")

        self.data = data if data is not None else {}
        self.data_encoded = None

        self.case_identifier = case_identifier
        self.activity_identifier = activity_identifier
        self.timestamp_identifier = timestamp_identifier

        self.attribute_identifiers = None

        self.logger.info("Recoding sequence data types...")
        self.recode_types()

        self.logger.info("Data initialization completed!")

    @classmethod
    def from_csv(cls, load_path: os.PathLike, read_params: dict = dict(), **kwargs) -> ECDataset:
        """Initialize ECDataset from a .csv file

        Args:
            load_path (os.PathLike): the path to the .csv file

        Returns:
            ECDataset: the new ECDataset object
        """
        data = pd.read_csv(load_path, **read_params)
        return cls(data=data, **kwargs)

    @classmethod
    def from_obj(cls, obj: ECDataset, **kwargs) -> ECDataset:
        """Initialize ECDataset from an existing ECDataset object and update parameters with given kwargs

        Args:
            obj (ECDataset): the ECDataset we want to initialize from

        Raises:
            TypeError: if an object from a different class is given as obj

        Returns:
            ECDataset: the new ECDataset object
        """

        if not isinstance(obj, cls):
            raise TypeError("object must be an instance of ECDataset")
        new_init_params = copy.deepcopy(vars(obj))
        for param, value in kwargs.items():
            new_init_params[param] = value

        return cls(**new_init_params)

    def setup_encoders(self, encoding_params):

        self.encoders = dict()
        for type, cols in encoding_params.items():
            
            encoder = type.split('_')[-1]
            if encoder == 'OneHotEncoder':
                encoder_obj = EncodingFactory.create(encoder, sparse_output=False, handle_unknown='ignore')
            else:
                encoder_obj = EncodingFactory.create(encoder)

            self.encoders.update({encoder: (encoder_obj, cols)})

    def encode(self, already_fitted: bool = False):
        
        data_to_encode = self.data

        transforms = list()
        encode_cols = list()
        
        if not already_fitted:  # training encoding
            for _, (encoder, cols) in self.encoders.items():
                if len(cols) > 0:
                    encode_cols.extend(cols)
                    encoder.fit(data_to_encode[cols])
                    transformed_cols = encoder.transform(data_to_encode[cols])
                    transforms.append(transformed_cols)
        else:                       # test encoding with train encoders
            for _, (encoder, cols) in self.encoders.items():
                if len(cols) > 0:
                    transformed_cols = encoder.transform(data_to_encode[cols])
                    transforms.append(transformed_cols)
        encoded_data = np.hstack(transforms) # we stack transformed cols in order of encoder cols in the data_config
        self.data_encoded = encoded_data
        self.attribute_identifiers = encode_cols

    def setup_transformers(self, transform_params):

        self.transformers = dict()
        for type, cols in transform_params.items():
            
            transformer = type.split('_')[-1]
            if transformer == 'PowerTransformer':
                # we do not standardize - we encode afterwards
                transformer_obj = TransformFactory.create(transformer, method='yeo-johnson', standardize=False)
            else:
                transformer_obj = TransformFactory.create(transformer)

            self.transformers.update({transformer: (transformer_obj, cols)})

    def transform(self, already_fitted: bool = False):
        
        transforms = list()
        transform_cols = list()
        if not already_fitted:  # training transformations
            for _, (transformer, cols) in self.transformers.items():
                if len(cols) > 0:
                    transform_cols.extend(cols)
                    transformer.fit(self.data[cols])
                    transformed_cols = transformer.transform(self.data[cols])
                    transforms.append(transformed_cols)
        else:                       # test encoding with train transformers
            for _, (transformer, cols) in self.transformers.items():
                if len(cols) > 0:
                    transformed_cols = transformer.transform(self.data[cols])
                    transforms.append(transformed_cols)
        
        if len(transforms) == 0:
            self.data_transformed = self.data.copy()
        else:
            transformed_data = np.hstack(transforms) # we stack transformed cols in order of encoder cols in the data_config

            # retransform to pd.DataFrame so that we can follow up with encoding
            all_encode_cols = list()
            for (_, cols) in self.encoders.values():
                all_encode_cols.extend(cols)
                
            if len(all_encode_cols) > 0:
                data_to_transform = self.data.copy()
                all_transform_cols = list()
                for (_, cols) in self.transformers.values():
                    all_transform_cols.extend(cols)
                col_diff = set(all_transform_cols).difference(set(all_encode_cols))
                if len(col_diff):
                    raise ValueError(f"cols transformed that are not needed for encoding - {' ,'.join(col_diff)}")
                for idx, col in enumerate(all_transform_cols):
                    data_to_transform[col] = transformed_data[:, idx]
                self.data_transformed = data_to_transform
            else:
                self.data_transformed = transformed_data
            # self.attribute_identifiers = encode_cols

    def recode_types(self) -> pd.DataFrame:
        """Recodes the types of the timestamp (-> datetime) and activity (str) columns

        Returns:
            pd.DataFrame: the pd.DataFrame with recoded column dtypes
        """
        recoded_times = pd.to_datetime(
            self.data[self.timestamp_identifier], format="mixed"
        )

        self.data.loc[:, self.timestamp_identifier] = recoded_times
        self.data.loc[:, self.activity_identifier] = self.data.loc[
            :, self.activity_identifier
        ].astype(object)
        self.data.loc[:, self.activity_identifier] = self.data.loc[
            :, self.activity_identifier
        ].astype(str)

    def pad_columns(
        self,
        cols_to_pad: list[str],
        n_pad: int = 1,
        forward_pad: str | None = "START",
        backward_pad: str | None = "END",
    ) -> pd.DataFrame:
        """Pads the existing sequence data with specified START and/or END tokens

        Args:
            cols_to_pad (list[str]): the columns to pad
            n_pad (int, optional): the length of the padding. Defaults to 1.
            forward_pad (str | None, optional): the pad token string for the START pad.
            None results in no padding. Defaults to 'START'.
            backward_pad (str | None, optional): the pad token string for the END pad.
            None results in no padding. Defaults to 'END'.

        Returns:
            pd.DataFrame: the padded pd.DataFrame
        """
        self.logger.info("Padding columns...")
        self.start_token = forward_pad
        self.end_token = backward_pad

        padded_data = self.data.copy()
        previous_dtypes = padded_data.dtypes

        padded_data = padded_data.groupby(self.case_identifier).apply(
            lambda x: _inner_pad(
                x,
                case_identifier=self.case_identifier,
                cols_to_pad=cols_to_pad,
                n_pad=n_pad,
                forward_pad=forward_pad,
                backward_pad=backward_pad,
            )
        )

        padded_data = padded_data.reset_index(drop=True)
        for col, dtype in enumerate(previous_dtypes):
            try:
                padded_data[padded_data.columns[col]] = padded_data.iloc(axis=1)[
                    col
                ].astype(dtype)
            except pd.errors.IntCastingNaNError:
                pass

        self.data = padded_data
        self.logger.info("Column padding completed!")

    def encode_activities(self, act_encoder: LabelEncoder = None):
        
        if act_encoder is None:
            act_encoder = LabelEncoder()
            act_encoder.fit(self.data[self.activity_identifier])
            self.act_encoder = act_encoder

        self.act_mapping = dict(
            zip(act_encoder.classes_, act_encoder.transform(act_encoder.classes_))
        )
        encoded_activities = (
            self.data[self.activity_identifier]
            .map(self.act_mapping)
            .fillna(-1)
            .astype(int)
        )
        self.data[self.activity_identifier] = encoded_activities

        self.start_activity = self.act_mapping[self.start_token]
        self.end_activity = self.act_mapping[self.end_token]

    def generate_time_features(self, features: list[str] = ["tsmn", "tscs", "tsle"]):

        features = [feature.lower() for feature in features]
        feature_frame = pd.DataFrame()

        for feature in features:
            try:
                feature_func = globals()[f"_calculate_{feature}"]
                self.logger.info(
                    f"Generating time feature information: {feature.upper()}"
                )
                feature_series = self.data.groupby(self.case_identifier).apply(
                    lambda x: feature_func(
                        x,
                        timestamp_identifier=self.timestamp_identifier,
                    ),
                    include_groups=False,
                )
                feature_frame = pd.concat(
                    [feature_frame, feature_series], axis=1
                )
            except KeyError as e:
                e.add_note(f"no function for feature '{feature}' implemented")
                raise
        feature_frame.index = [idx[1] for idx in feature_frame.index]

        self.data = pd.concat([self.data, feature_frame], axis=1)

    def train_test_split(
        self, train_pct: float, cv: int = 1
    ) -> tuple[ECDataset, ECDataset] | list[tuple[ECDataset, ECDataset]]:
        """Function for splitting an existing SequenceData object into two distinct SequenceData objects (train and test).

        Args:
            train_pct (float): Percentage share of the data that should be transferred into the training set object.
            The test set consists of the remaining sequences.
            cv (int, optional): If k-fold cross-validation should be performed where cv is the number of folds.
            Defaults to False.

        Returns:
            tuple[ECDataset]: training and test instances of the corresponding sequences as ECDataset objects
        """
        if cv > 1:  # generate k-fold cross validation datasets
            self.logger.info(
                f"Splitting train and test data ({1/cv*100:.0f}-{(1-(1/cv))*100:.0f} split with {cv} folds)"
            )
            # split the data into cv folds
            all_ids = self.data[self.case_identifier].unique()
            np.random.shuffle(all_ids)
            id_folds = [id_fold for id_fold in _batch_samples(all_ids, cv)]
            folds = list()
            for idf_idx in range(0, len(id_folds)):
                test_ids = id_folds[idf_idx].tolist()
                train_ids = list(
                    itertools.chain(
                        *[id_folds[i] for i in range(0, len(id_folds)) if i != idf_idx]
                    )
                )
                train_data = self.data[self.data[self.case_identifier].isin(train_ids)]
                test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

                train_data_obj = ECDataset.from_obj(self, data=train_data)
                test_data_obj = ECDataset.from_obj(self, data=test_data)
                folds.append(tuple([train_data_obj, test_data_obj]))

            return folds
        else:
            self.logger.info(
                f"Splitting train and test data ({train_pct*100:.0f}-{(1-train_pct)*100:.0f} split)"
            )
            all_ids = self.data[self.case_identifier].unique()
            train_ids = np.random.choice(
                a=all_ids, size=int(len(all_ids) * train_pct), replace=False
            )
            train_set = set(train_ids)
            test_ids = [id for id in all_ids if id not in train_set]

            train_data = self.data[self.data[self.case_identifier].isin(train_ids)]
            test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

            train_data_obj = ECDataset.from_obj(self, data=train_data)
            test_data_obj = ECDataset.from_obj(self, data=test_data)

            return train_data_obj, test_data_obj

    def get_characteristics(self) -> dict:

        log_characteristics = dict()

        log_characteristics["n_cases"] = self._get_n_cases()
        log_characteristics["n_events"] = self._get_n_events()
        log_characteristics["n_activities"] = self._get_n_activities()
        log_characteristics["mean_trace_len"] = float(self._get_mean_trace_len())
        log_characteristics["median_trace_len"] = float(self._get_median_trace_len())
        log_characteristics["max_trace_len"] = self._get_max_trace_len()

        return log_characteristics

    def _get_n_cases(self) -> int:
        n_cases = self.data[self.case_identifier].nunique()
        return n_cases

    def _get_n_events(self) -> int:
        n_events = len(self.data)
        return n_events

    def _get_n_activities(self) -> int:
        n_activities = self.data[self.activity_identifier].nunique()
        return n_activities

    def _get_median_trace_len(self) -> float:
        trace_lens = self._get_trace_lens()
        trace_lens.sort()
        center_idx = int(len(trace_lens) / 2)
        if len(trace_lens) % 2 == 0:
            median_trace_len = sum(trace_lens[center_idx - 1 : center_idx + 1]) / 2
        else:
            median_trace_len = trace_lens[center_idx]
        return median_trace_len

    def _get_mean_trace_len(self) -> float:
        mean_trace_len = self._get_trace_lens().mean()
        return mean_trace_len

    def _get_max_trace_len(self) -> float:
        max_trace_len = int(self._get_trace_lens().max())
        return max_trace_len

    def _get_trace_lens(self) -> np.array[int]:
        trace_lens = np.array(self.data[self.case_identifier].value_counts())
        return trace_lens
    
    def to_sequence_data(self) -> SequenceData:

        sequence_data_obj = None

        return sequence_data_obj

def _calculate_tsmn(df: pd.DataFrame, timestamp_identifier: str):
    tsmn = pd.Series(
        [
            ts.hour * 60 * 60 + ts.minute * 60 + ts.second
            for ts in df[timestamp_identifier]
        ], index=df.index
    )
    tsmn.name = "tsmn"
    return tsmn


def _calculate_tscs(df: pd.DataFrame, timestamp_identifier: str):
    start_time = min(df[timestamp_identifier])
    difference = pd.to_datetime(df[timestamp_identifier]) - start_time
    tscs = difference.apply(dt.timedelta.total_seconds).astype(int)
    tscs.name = "tscs"
    return tscs


def _calculate_tsle(df: pd.DataFrame, timestamp_identifier: str):
    if len(df) > 1:
        time_lagged = pd.to_datetime(df[timestamp_identifier]).shift(1).bfill()
        difference = pd.to_datetime(df[timestamp_identifier]) - time_lagged
        tsle = difference.apply(dt.timedelta.total_seconds).astype(int)
    else:
        tsle = (pd.to_datetime(df[timestamp_identifier]) - pd.to_datetime(df[timestamp_identifier])).apply(dt.timedelta.total_seconds).astype(int)
    tsle.name = "tsle"
    return tsle


def _inner_pad(
    df: pd.DataFrame,
    case_identifier: str,
    cols_to_pad: list[str],
    n_pad: int = 1,
    forward_pad: str | None = "START",
    backward_pad: str | None = "END",
) -> pd.DataFrame:

    if forward_pad:
        df = (
            df.reset_index(drop=True)
            .reindex(range(-n_pad, len(df)))
            .reset_index(drop=True)
        )
        for col in cols_to_pad:
            dtype = df.loc(axis=1)[col].dtype
            if dtype in ['float', 'int']:
                df.loc[0 : n_pad - 1, col] = 0
            else:
                df.loc[0 : n_pad - 1, col] = forward_pad
        
        df[case_identifier] = df[case_identifier].bfill()
    if backward_pad:
        df = (
            df.reset_index(drop=True)
            .reindex(range(0, len(df) + n_pad))
            .reset_index(drop=True)
        )
        for col in cols_to_pad:
            dtype = df.loc(axis=1)[col].dtype
            if dtype in ['float', 'int']:
                df.loc[len(df) - n_pad :, col] = 0
            else:
                df.loc[len(df) - n_pad :, col] = backward_pad
        df[case_identifier] = df[case_identifier].ffill()

    return df


def _batch_samples(samples, nbatches: int):
    nsamples = len(samples)
    batchsize = math.ceil(nsamples / nbatches)
    for idx in range(0, nsamples, batchsize):
        yield samples[idx : min(idx + batchsize, nsamples)]
