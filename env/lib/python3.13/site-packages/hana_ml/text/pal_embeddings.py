"""
Embeddings inside HANA

    * :class:`PALEmbeddings`
"""

#pylint: disable=redefined-builtin
#pylint: disable=bare-except
import uuid
import logging
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    arg,
    try_drop
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class PALEmbeddingsBase(PALBase):
    """
    PAL embeddings base class.
    """
    def __init__(self, model_version=None, max_token_num=None, pca_dim_num=None):
        super(PALEmbeddingsBase, self).__init__()
        self.result_ = None
        self.connection_context = None
        self.embedding_col = None
        self.target = None
        self.model_version = arg('model_version', model_version, str)
        self.max_token_num = arg('max_token_num', max_token_num, int)
        self.pca_dim_num = arg('pca_dim_num', pca_dim_num, int)

    def _fit_transform(self, data, key, target, thread_number=None, batch_size=None, is_query=None, max_token_num=None):
        """
        Predict the embeddings.

        Parameters
        ----------
        data: DataFrame
            Data.
        key: str
            Key.
        target: str
            Target.
        thread_number: int
            Thread number.
        batch_size: int
            Batch size.
        is_query: bool
            Use different embedding model for query purpose.
        max_token_num: int
            Maximum number of tokens.
        """
        conn = data.connection_context
        self.connection_context = conn
        self.target = target
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        embeddings_tbl = '#PAL_EMBEDDINGS_RESULT_TBL_{}_{}'.format(0, unique_id)
        stats_tbl = '#PAL_EMBEDDINGS_STATS_TBL_{}_{}'.format(0, unique_id)
        outputs = [embeddings_tbl, stats_tbl]
        param_rows = [("IS_QUERY", is_query, None, None),
                      ("BATCH_SIZE", batch_size, None, None),
                      ("MODEL_VERSION", None, None, self.model_version),
                      ("THREAD_NUMBER", thread_number, None, None),
                      ("MAX_TOKEN_NUM", max_token_num if max_token_num else self.max_token_num, None, None),
                      ("PCA_DIM_NUM", self.pca_dim_num, None, None)]
        data_ = data.select([key, target])
        try:
            self._call_pal_auto(conn,
                                'PAL_TEXTEMBEDDING',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, outputs)
            raise
        except Exception as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        return conn.table(embeddings_tbl), conn.table(stats_tbl)

class PALEmbeddings(PALEmbeddingsBase):
    """
    Embeds input documents into vectors.

    Parameters
    ----------
    model_version: {'SAP_NEB.20240715', 'SAP_GXY.20250407'}, optional
        Model version to use. If None, defaults to 'SAP_NEB.20240715'.

        Options:

        - 'SAP_NEB.20240715'
        - 'SAP_GXY.20250407'

        Defaults to None (uses 'SAP_NEB.20240715' by default).
    max_token_num: int, optional
        Maximum number of tokens per document depends on the embedding model.

    pca_dim_num: int, optional
        If set, applies PCA to reduce the dimensionality of the embeddings to the specified number.

    Attributes
    ----------
    result_ : DataFrame
        The embedding result.

    stat_ : DataFrame
        The statistics.

    Examples
    --------
    Suppose you have a HANA DataFrame `df` with columns 'ID' and 'TEXT'.
    To embed the documents into vectors, create a PALEmbeddings instance and call `fit_transform`:

    >>> from hana_ml.text.pal_embeddings import PALEmbeddings
    >>> embedder = PALEmbeddings(model_version='SAP_GXY.20250407')
    >>> result = embedder.fit_transform(data=df, key='ID', target='TEXT')
    >>> # The result is a DataFrame with the original data and embedding columns
    >>> print(result.collect())

    You can also embed multiple text columns at once if you have more than one text column:

    >>> embedder = PALEmbeddings(model_version='SAP_GXY.20250407')
    >>> result = embedder.fit_transform(data=df, key='ID', target=['TEXT1', 'TEXT2'])
    >>> print(result.collect())
    """
    def __init__(self, model_version=None, max_token_num=None, pca_dim_num=None):
        super(PALEmbeddings, self).__init__(model_version=model_version, max_token_num=max_token_num, pca_dim_num=pca_dim_num)

    def fit_transform(self, data, key, target, thread_number=None, batch_size=None, is_query=None, max_token_num=None):
        """
        Embed input documents into vectors.

        Parameters
        ----------
        data: DataFrame
            Input data containing the documents to embed.
        key: str
            Name of the key column.
        target: str or list of str
            Name(s) of the text column(s) to embed.
        thread_number: int, optional
            Number of HTTP connections to the backend embedding service (1-10).

            Defaults to 6.
        batch_size: int, optional
            Number of documents batched per request (1-50).

            Defaults to 10.
        is_query: bool, optional
            If True, use query embedding for Asymmetric Semantic Search.

            Defaults to False.
        max_token_num: int, optional
            Maximum number of tokens per document depends on the embedding model.

            - 'SAP_NEB.20240715': 1024 (default is 256 if not set)
            - 'SAP_GXY.20250407': 1024 (default is 512 if not set)

            If ``max_token_num`` is not set, the default value for the selected model version will be used.
            Defaults to None (uses the default value of the selected embedding model).

        Returns
        -------
        DataFrame
            DataFrame containing the original data and embedding columns.
        """
        thread_number = arg('thread_number', thread_number, int)
        batch_size = arg('batch_size', batch_size, int)
        is_query = arg('is_query', is_query, bool)
        max_token_num = arg('max_token_num', max_token_num, int)
        self.stat_ = None
        self.result_ = None
        if isinstance(target, (list, tuple)):
            for tar_col in target:
                result, stats = self._fit_transform(data, key, tar_col, thread_number=thread_number, batch_size=batch_size, is_query=is_query, max_token_num=max_token_num)
                result = result.select([result.columns[0], result.columns[1]])
                if self.result_ is None:
                    self.result_ = result.rename_columns({result.columns[1]: result.columns[1] + '_' + tar_col})
                    self.stat_ = {tar_col: stats}
                else:
                    result = result.rename_columns({result.columns[1]: result.columns[1] + '_' + tar_col})
                    self.result_ = self.result_.set_index(result.columns[0]).join(result.set_index(result.columns[0]))
                    self.stat_[tar_col] = stats
        else:
            result, stats = self._fit_transform(data, key, target, thread_number=thread_number, batch_size=batch_size, is_query=is_query, max_token_num=max_token_num)
            self.result_ = result
            self.stat_ = stats
        if not self._disable_hana_execution:
            self.result_ = data.set_index(key).join(self.result_.set_index(result.columns[0]))
        return self.result_
