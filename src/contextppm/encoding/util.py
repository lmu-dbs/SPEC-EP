from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
import numpy as np
import pandas as pd

class EncodingFactory:
    _encoding = {
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
        "MinMaxScaler": MinMaxScaler,
    }

    @classmethod
    def create(cls, encoding, **encoding_params):
        if encoding in cls._encoding:
            encoding_class = cls._encoding[encoding]
            return encoding_class(**encoding_params)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

class TransformFactory:
    _transform = {
        "PowerTransformer": PowerTransformer
    }

    @classmethod
    def create(cls, transform, **transform_params):
        if transform in cls._transform:
            transform_class = cls._transform[transform]
            return transform_class(**transform_params)
        else:
            raise ValueError(f"Unknown transform: {transform}")

class Decoding:

    def __init__(self, encoders: dict):
        self.encoders = encoders

    def decode_sample(self, encoded_sample) -> dict:

        cols_for_encoder = dict()
        for encoder, cols in self.encoders.values():
            if len(cols) > 0:
                if isinstance(encoder, OneHotEncoder):
                    ncols_cols_tpl = (sum([len(enc_cats) for enc_cats in encoder.categories_]), cols)
                elif isinstance(encoder, StandardScaler) or isinstance(encoder, RobustScaler) or isinstance(encoder, MinMaxScaler):
                    ncols_cols_tpl = (len(cols), cols)
                else:
                    raise NotImplementedError(f"encoder type {type(encoder)} not implemented for decoding")
                cols_for_encoder[encoder] = ncols_cols_tpl
            
        decoded_sample_dict = dict()

        start_index = 0
        for encoder, (n_decode_cols, decode_cols) in cols_for_encoder.items():
            if n_decode_cols > 0:
                sample_to_decode = encoded_sample[:, start_index:(start_index+n_decode_cols)]
                
                decoded_sample = encoder.inverse_transform(sample_to_decode)

                for dict_col, decoded_col in zip(decode_cols, decoded_sample[0,:]):
                    decoded_sample_dict[dict_col] = decoded_col

                start_index += n_decode_cols

        return decoded_sample_dict
    
    def decode_samples(self, encoded_samples) -> dict:

        decoded_samples = [self.decode_sample(sample) for sample in encoded_samples]

        # transform list of single value dicts to dict of lists per encoded attribute
        decoded_samples_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in decoded_samples[0].keys():
            decoded_samples_list_dict[attribute] = [sample[attribute] for sample in decoded_samples]

        return decoded_samples_list_dict
    
    def decode_sample_sequences(self, encoded_sample_sequences) -> dict:

        decoded_sample_sequences = [self.decode_samples(sample_seq) for sample_seq in encoded_sample_sequences]

        # transform list of single value dicts to dict of lists per encoded attribute
        decoded_sample_sequences_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in decoded_sample_sequences[0].keys():
            decoded_sample_sequences_list_dict[attribute] = [sequence[attribute] for sequence in decoded_sample_sequences]

        return decoded_sample_sequences_list_dict


class Retransformation:

    def __init__(self, transformers: dict):
        self.transformers = transformers

    def retransform_sample(self, transformed_sample) -> dict:

        cols_for_transformer = dict()
        for transformer, cols in self.transformers.values():
            if len(cols) > 0:
                if isinstance(transformer, PowerTransformer):
                    ncols_cols_tpl = (len(cols), cols)
                else:
                    raise NotImplementedError(f"transformer type {type(transformer)} not implemented for decoding")
                cols_for_transformer[transformer] = ncols_cols_tpl
            
        retransformed_sample_dict = dict()

        start_index = 0
        for transformer, (n_retransform_cols, retransform_cols) in cols_for_transformer.items():
            if n_retransform_cols > 0:
                sample_to_retransform = transformed_sample[:, start_index:(start_index+n_retransform_cols)]
                
                decoded_sample = transformer.inverse_transform(pd.DataFrame(sample_to_retransform, columns=retransform_cols))

                for dict_col, decoded_col in zip(retransform_cols, decoded_sample[0,:]):
                    retransformed_sample_dict[dict_col] = decoded_col

                start_index += n_retransform_cols

        return retransformed_sample_dict
    
    def retransform_samples(self, transformed_samples) -> dict:

        if isinstance(transformed_samples, dict):
            rearranged_transformed_samples = self.rearrange_samples(transformed_samples)
        else:
            rearranged_transformed_samples = transformed_samples.copy()

        retransformed_samples = [self.retransform_sample(sample) for sample in rearranged_transformed_samples]

        # transform list of single value dicts to dict of lists per encoded attribute
        retransformed_samples_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in retransformed_samples[0].keys():
            retransformed_samples_list_dict[attribute] = [sample[attribute] for sample in retransformed_samples]

        if isinstance(transformed_samples, dict):
            transformed_samples.update(retransformed_samples_list_dict)
            retransformed_samples_list_dict = transformed_samples

        return retransformed_samples_list_dict
    
    def rearrange_samples(self, transformed_samples) -> list:
        all_transform_cols = list()
        for (_, cols) in self.transformers.values():
            all_transform_cols.extend(cols)
        transformed_sample_cols = list()
        for col in all_transform_cols:
            transformed_sample_cols.append(np.array(transformed_samples[col]))
        rearranged_transformed_samples = np.stack(transformed_sample_cols).transpose()
        rearranged_transformed_samples = [np.array([sample]) for sample in rearranged_transformed_samples]

        return rearranged_transformed_samples

    def rearrange_sample_sequences(self, transformed_sample_sequences) -> list:
        all_transform_cols = list()
        for (_, cols) in self.transformers.values():
            all_transform_cols.extend(cols)
        rearranged_transformed_sample_sequences = list()
        for sequence_idx in range(len(transformed_sample_sequences[all_transform_cols[0]])):
            rearranged_sample_sequence = list()
            for col in all_transform_cols:
                rearranged_sample_sequence.append(np.array(transformed_sample_sequences[col][sequence_idx]))
            rearranged_sample_sequence = np.stack(rearranged_sample_sequence).transpose()
            rearranged_transformed_sample_sequences.append([np.array([sample]) for sample in rearranged_sample_sequence])
        
        return rearranged_transformed_sample_sequences

    def retransform_sample_sequences(self, transformed_sample_sequences) -> dict:

        rearranged_transformed_sample_sequences = self.rearrange_sample_sequences(transformed_sample_sequences)
        
        retransformed_sample_sequences = [self.retransform_samples(sample_seq) for sample_seq in rearranged_transformed_sample_sequences]

        # transform list of single value dicts to dict of lists per encoded attribute
        retransformed_sample_sequences_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in retransformed_sample_sequences[0].keys():
            retransformed_sample_sequences_list_dict[attribute] = [sequence[attribute] for sequence in retransformed_sample_sequences]

        transformed_sample_sequences.update(retransformed_sample_sequences_list_dict)
        retransformed_sample_sequences_list_dict = transformed_sample_sequences

        return retransformed_sample_sequences_list_dict