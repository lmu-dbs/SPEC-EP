import numpy as np
from ..dataset.ecdataset import ECDataset
from ..util.logging import init_logging
from .util import ECFactory

logger = init_logging(__name__, "pcc.log")

class ProcessContextClustering():

    """Clustering model working on encoded process event context information to produce clusters of context information
    for subsequent use in Predictive Process Monitoring (PPM) tasks.
    """

    def __init__(self, encoding_params: dict, transform_params: dict, clustering_type, cluster_params: dict):
        
        logger.info(f'Initializing clustering model - { {k:v for k,v in encoding_params.items()} } - { {k:v for k,v in transform_params.items()} }')
        
        self.data_train = None
        self.data_test = None
        self.encoders = None
        self.encoding_params = encoding_params
        self.transformers = None
        self.transform_params = transform_params

        self.clustering_type = clustering_type
        self.cluster_params = cluster_params
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves clustering of the context data for
        subsequent prediction of the cluster memberships used in PPM tasks
        """
        self.event_clustering = ECFactory.create(
            clustering=self.clustering_type,
            **self.cluster_params,
        )

        self.event_clustering.fit(X=self.data_train.data_encoded)

    def predict(self) -> np.array:
        """Generates predictions of cluster memberships given context data."""

        logger.info(f'Predicting cluster memberships')

        train_clusters = self.event_clustering.predict(self.data_train.data_encoded)
        test_clusters = self.event_clustering.predict(self.data_test.data_encoded)

        self.data_train.data["context_cluster"] = train_clusters
        self.data_test.data["context_cluster"] = test_clusters

        # append cluster memberships to activity identifier
        train_agg = self.data_train.data.agg(lambda x: f"{x[self.data_train.activity_identifier]}_context_cluster_{x['context_cluster']}", axis=1)
        test_agg = self.data_test.data.agg(lambda x: f"{x[self.data_test.activity_identifier]}_context_cluster_{x['context_cluster']}", axis=1)
        self.data_train.data[f"{self.data_train.activity_identifier}_context"] = train_agg 
        self.data_test.data[f"{self.data_test.activity_identifier}_context"] = test_agg


    def load_data(self, train: ECDataset, test: ECDataset):
        self.data_train = train
        self.data_test = test

    def prepare_train(self, additional_cols_to_pad: list[str]) -> None:
        
        logger.info('Preparing training data for clustering...')
        if self.data_train is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')

        # # padding trace data by one START and one END token
        # self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier] + additional_cols_to_pad, n_pad=1)
        
        # forward and backward fill timestamp column
        
        # CAUTION
        # this is dangerous in case we do not have full timestamp information!
        # we would peak into the future timestamps if we have NAN timestamps in the middle of traces
        self.data_train.data.loc[:, self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data.loc[:, self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()
        
        # generating time features
        # TSLE: time since last event
        # TSMN: time since midnight
        # TSCS: time since case start
        self.data_train.generate_time_features(["tsle", "tsmn", "tscs"])

        self.train_max_trace_len = self.data_train._get_max_trace_len()
        
        # setup encoding and transformers
        self.data_train.setup_encoders(self.encoding_params)
        # self.data_train.setup_transformers(self.transform_params)

        # transform
        # self.data_train.transform()
        
        # encode
        self.data_train.encode()

        logger.info('Training data prepared!')

    def prepare_test(self, additional_cols_to_pad: list[str]) -> None:
        
        logger.info('Preparing test data for clustering...')
        if self.data_test is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        # # padding trace data by one START and one END token
        # self.data_test.pad_columns(cols_to_pad=[self.data_train.activity_identifier] + additional_cols_to_pad, n_pad=1)

        # forward and backward fill timestamp column

        # CAUTION
        # this is dangerous in case we do not have full timestamp information!
        # we would peak into the future timestamps if we have NAN timestamps in the middle of traces
        self.data_test.data.loc[:, self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].ffill()
        self.data_test.data.loc[:, self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].bfill()

        # generating time features
        # TSLE: time since last event
        # TSMN: time since midnight
        # TSCS: time since case start
        self.data_test.generate_time_features(["tsle", "tsmn", "tscs"])
        
        # self.data_test.transformers = self.data_train.transformers
        self.data_test.encoders = self.data_train.encoders

        # transform
        # self.data_test.transform(already_fitted=True)

        # encode
        self.data_test.encode(already_fitted=True)
        
        logger.info('Test data prepared!')