"""
This module contains Python wrapper for SAP HANA PAL Unified Time Series.

The following classes are available:

    * :class:`UnifiedTimeSeries`
    * :class:`MassiveUnifiedTimeSeries`
"""
#pylint: disable=too-many-lines, too-many-branches, unused-argument, use-a-generator
#pylint: disable=line-too-long, too-many-statements, consider-using-dict-items
#pylint: disable=too-many-locals, too-many-instance-attributes, c-extension-no-member
#pylint: disable=too-many-arguments, invalid-name, unnecessary-pass, too-many-nested-blocks
#pylint: disable=ungrouped-imports, bare-except, super-with-arguments
#pylint: disable=consider-using-f-string, attribute-defined-outside-init
#pylint: disable=broad-except, no-member, access-member-before-definition, too-many-positional-arguments
import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_base import try_drop
from hana_ml.ml_exceptions import FitIncompleteError
from .tsa.arima import _col_index_check
from .utility import _map_param, check_pal_function_exist
from .sqlgen import trace_sql
from .pal_base import (
    arg,
    PALBase,
    ParameterTable,
    require_pal_usable,
    pal_param_register,
    ListOfTuples,
    ListOfStrings)
logger = logging.getLogger(__name__)

def _params_check(input_dict, param_map, func):
    update_params = {}
    if not input_dict or input_dict is None:
        return update_params

    for parm in input_dict:
        if parm in ['categorical_variable']:
            pass
        if parm in param_map.keys():
            parm_val = input_dict[parm]
            arg_map = param_map[parm]
            if arg_map[1] == ListOfStrings and isinstance(parm_val, str):
                parm_val = [parm_val]
            if len(arg_map) == 2:
                update_params[arg_map[0]] = (arg(parm, parm_val, arg_map[1]), arg_map[1])
            else:
                update_params[arg_map[0]] = (arg(parm, parm_val, arg_map[2]), arg_map[1])
        else:
            err_msg = f'"{parm}" is not a valid parameter name for initializing a {func} model!'
            logger.error(err_msg)
            raise KeyError(err_msg)

    return update_params

func_dict = {
    'amtsa': 0,
    'arima': 1,
    'bsts': 2,
    'smooth': 3
}

_AMTSA_param = {
    'growth' : ('GROWTH', str),
    'logistic_growth_capacity' : ('CAP', float),
    'seasonality_mode'  : ('SEASONALITY_MODE', str),
    'seasonality'  : ('SEASONALITY', ListOfStrings),
    'num_changepoints' : ('NUM_CHANGEPOINTS', int),
    'changepoint_range' : ('CHANGEPOINT_RANGE', float),
    'regressor' : ('REGRESSOR', ListOfStrings),
    'changepoints' : ('CHANGE_POINT', ListOfStrings),
    'yearly_seasonality' : ('YEARLY_SEASONALITY', int, {'auto': -1, 'false': 0, 'true': 1}),
    'weekly_seasonality' : ('WEEKLY_SEASONALITY', int, {'auto': -1, 'false': 0, 'true': 1}),
    'daily_seasonality' : ('DAILY_SEASONALITY', int, {'auto': -1, 'false': 0, 'true': 1}),
    'seasonality_prior_scale' : ('SEASONALITY_PRIOR_SCALE', float),
    'holiday_prior_scale' : ('HOLIDAYS_PRIOR_SCALE', float),
    'changepoint_prior_scale' : ('CHANGEPOINT_PRIOR_SCALE', float),
    'target_type': ('TARGET_TYPE', str),
    'start_point': ('START_POINT', str),
    'interval': ('INTERVAL', int),
    'holiday': ('HOLIDAY', str)
    }

_AMTSA_predict_param = {
    'logistic_growth_capacity' : ('CAP', float),
    'interval_width' : ('INTERVAL_WIDTH', float),
    'uncertainty_samples' : ('UNCERTAINTY_SAMPLES', int),
    'show_explainer' : ('EXPLAINER', bool),
    'decompose_seasonality' : ('EXPLAIN_SEASONALITY', bool),
    'decompose_holiday' : ('EXPLAIN_HOLIDAY', bool)}

_ARIMA_param = {
    'seasonal_period' : ('SEASONAL_PERIOD', int),
    'seasonality_criterion' : ('SEASONALITY_CRITERION', float),
    'd' : ('D',int),
    'kpss_significance_level'  : ('KPSS_SIGNIFICANCE_LEVEL', float),
    'max_d'  : ('MAX_D', int),
    'seasonal_d' : ('SEASONAL_D', int),
    'ch_significance_level' : ('CH_SIGNIFICANCE_LEVEL', float),
    'max_seasonal_d' : ('MAX_SEASONAL_D', int),
    'max_p' : ('MAX_P', int),
    'max_q' : ('MAX_Q', int),
    'max_seasonal_p'  : ('MAX_SEASONAL_P', int),
    'max_seasonal_q'  : ('MAX_SEASONAL_Q', int),
    'information_criterion' : ('INFORMATION_CRITERION', int, {'aicc': 0, 'aic': 1, 'bic': 2}), # based on latest version of autoARIMA
    'search_strategy' : ('SEARCH_STRATEGY', int, {'exhaustive': 0, 'stepwise': 1}), # based on latest version of autoARIMA
    'max_order' : ('MAX_ORDER', int),
    'initial_p' : ('INITIAL_P', int),
    'initial_q' : ('INITIAL_Q', int),
    'initial_seasonal_p' : ('INITIAL_SEASONAL_P', int),
    'initial_seasonal_q'  : ('INITIAL_SEASONAL_Q', int),
    'guess_states'  : ('GUESS_STATES', int),
    'max_search_iterations' : ('MAX_SEARCH_ITERATIONS', int),
    'method' : ('METHOD', int, {'css':0, 'mle':1, 'css-mle':2}),
    'allow_linear' : ('ALLOW_LINEAR', bool), # based on latest version of autoARIMA
    'forecast_method'  : ('FORECAST_METHOD', int, {'formula_forecast':0, 'innovations_algorithm':1}),
    'output_fitted'  : ('OUTPUT_FITTED', bool),
    'thread_ratio' : ('THREAD_RATIO', float),
    'background_size' : ('BACKGROUND_SIZE', int)}

_ARIMA_predict_param = {
    'forecast_method' : ('FORECAST_METHOD', int, {'formula_forecast':0,
                                                    'innovations_algorithm':1,
                                                    'truncation_algorithm':2}),
    'forecast_length' : ('FORECAST_LENGTH', int),
    'thread_ratio' : ('THREAD_RATIO', float),
    'top_k_attributions' : ('TOP_K_ATTRIBUTIONS', int),
    'trend_mod' : ('TREND_MOD', float),
    'trend_width' : ('TREND_WIDTH', float),
    'seasonal_width' : ('SEASONAL_WIDTH', float)}

_BSTS_param = {
    'burn': ('BURN', float),
    'niter': ('NITER', int),
    'seasonal_period': ('SEASONAL_PERIOD', int),
    'expected_model_size': ('EXPECTED_MODEL_SIZE', int),
    'seed': ('SEED', int)}

_BSTS_predict_param = {
    'horizon': ('HORIZON', int)}

_SMOOTH_param = {
    'model_selection': ('MODELSELECTION', bool),
    'forecast_model_name': ('FORECAST_MODEL_NAME', str),
    'optimizer_time_budget': ('OPTIMIZER_TIME_BUDGET', int),
    'max_iter': ('MAX_ITERATION', int),
    'optimizer_random_seed': ('OPTIMIZER_RANDOM_SEED', int),
    'thread_ratio': ('THREAD_RATIO', float),
    'expost_flag': ('EXPOST_FLAG', bool),
    'alpha': ('ALPHA', float),
    'beta': ('BETA', float),
    'gamma': ('GAMMA', float),
    'phi': ('PHI', float),
    'forecast_num': ('FORECAST_NUM', int),
    'seasonal_period': ('CYCLE', int),
    'seasonal': ('SEASONAL', int, {'multiplicative': 0, 'additive': 1}), # based on latest version of auto exponential smoothing
    'initial_method': ('INITIAL_METHOD', int),
    'training_ratio': ('TRAINING_RATIO', float),
    'damped': ('DAMPED', bool),
    'accuracy_measure': ('ACCURACY_MEASURE', str),
    'seasonality_criterion': ('SEASONALITY_CRITERION', float),
    'trend_test_method': ('TREND_TEST_METHOD', int, {'mk': 1, 'difference-sign': 2}), # based on latest version of auto exponential smoothing
    'trend_test_alpha': ('TREND_TEST_ALPHA', float),
    'alpha_min': ('ALPHA_MIN', float),
    'beta_min': ('BETA_MIN', float),
    'gamma_min': ('GAMMA_MIN', float),
    'phi_min': ('PHI_MIN', float),
    'alpha_max': ('ALPHA_MAX', float),
    'beta_max': ('BETA_MAX', float),
    'gamma_max': ('GAMMA_MAX', float),
    'phi_max': ('PHI_MAX', float),
    #'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float), # this param is reserved
    #'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float), # this param is reserved
    'level_start': ('LEVEL_START', float),
    'trend_start': ('TREND_START', float),
    'season_start': ('SEASON_START', ListOfTuples),
    'decom_state': ('DECOM_STATE', bool)}

map_dict = {'AMTSA': _AMTSA_param,
            'ARIMA': _ARIMA_param,
            'BSTS': _BSTS_param,
            'SMOOTH': _SMOOTH_param} # no predict param

map_dict_predict = {'AMTSA': _AMTSA_predict_param,
                    'ARIMA': _ARIMA_predict_param,
                    'BSTS': _BSTS_predict_param}

class UnifiedTimeSeries(PALBase):
    """
    The Python wrapper for SAP HANA PAL Unified Time Series function.
    The Unified Time Series algorithms include:

    - Additive Model Time Series Analysis (AMTSA)
    - Auto Regressive Integrated Moving Average (ARIMA)
    - Bayesian Structural Time Series (BSTS)
    - Exponential Smoothing (SMOOTH)


    Parameters
    ----------

    func : str

        The name of a specified time series algorithm.

        The following algorithms are supported:

        - 'AMTSA': Additive Model Time Series Analysis
        - 'ARIMA': Auto Regressive Integrated Moving Average
        - 'BSTS': Bayesian Structural Time Series
        - 'SMOOTH': Auto Exponential Smoothing

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

        - **'AMTSA'** : :class:`~hana_ml.algorithms.pal.tsa.additive_model_time_series.AdditiveModelTimeSeriesAnalysis`
          AMTSA in UnifiedTimeSeries has some additional parameters, please see the following section.

          - target_type : str, optional, specify the type when converting the INT type of key column to other types.
          - start_point : str, optional, specify a start point in type conversion.
          - interval : int, optional, specify an interval in type conversion.
          - holiday : str, optional, add holiday to model in a json format, including name, timestamp, (optional) lower_window, and (optional) upper_window elements. For example: '{ "name": "New Year", "timestamp": "2025-01-01" }'.

        - **'ARIMA'** : :class:`~hana_ml.algorithms.pal.tsa.auto_arima.AutoARIMA`

        - **'BSTS'** : :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.TripleExponentialSmoothing`

        - **'SMOOTH'** : :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.AutoExponentialSmoothing`

        For more parameter mappings of hana_ml and HANA PAL, please refer to the doc page: :ref:`param_mapping`

    Attributes
    ----------

    model_ : DataFrame
        Model information.

    statistics_ : DataFrame
        Statistics.

    decompose_ : DataFrame
        Decomposition values.


    Examples
    --------
    >>> uts = UnifiedTimeSeries(func='AMTSA')

    Perform fit_predict():

    >>> uts.fit_predict(data=df)

    Output:

    >>> uts.model_.collect()
    >>> uts.statistics_.collect()
    >>> uts.decompose_.collect()

    """

    def __init__(self,
                 func,
                 **kwargs):
        super(UnifiedTimeSeries, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.func = self._arg('Function name', func, func_dict)
        #print(f"self.func: {self.func}")
        self.func_name = func.upper()
        #print(f"self.func_name: {self.func_name}")
        func_map = map_dict[self.func_name]
        self.__pal_params = {}
        self.params = {**kwargs}
        if not self._disable_hana_execution:
            self.__pal_params = _params_check(input_dict=self.params,
                                              param_map=func_map,
                                              func=self.func_name)
        self.model_ = None
        self.statistics_ = None
        self.fit_data = None

    @trace_sql
    def fit(self,
            data,
            key=None,
            endog=None,
            exog=None):
        """
        Fit function.

        Parameters
        ----------
        data : DataFrame
            Training data.

        key : str, optional

            Name of ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        exog : str or list of str, optional
            The column(s) of exogenous variables to be used in the model.

            Defaults to all non-ID, non-endog columns if not provided.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        setattr(self, 'hanaml_fit_params', pal_param_register())
        setattr(self, "training_data", data)
        setattr(self, 'key', key)
        setattr(self, 'endog', endog)
        setattr(self, 'exog', None)
        cols = data.columns
        index = data.index
        self.is_index_int = True

        key = self._arg('key', key, str)
        if index is not None:
            key = _col_index_check(key, 'key', index, cols)
        else:
            if key is None:
                key = cols[0]

        key = self._arg('key', key, str)
        if key is not None and key not in cols:
            msg = f"Please select key from {cols}!"
            logger.error(msg)
            raise ValueError(msg)
        cols.remove(key)

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = f"Please select endog from {cols}!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]
        cols.remove(endog)

        if self.func == 3: # SMOOTH does not support exog
            exog = None
        else: # not SMOOTH
            if exog is not None:
                if isinstance(exog, str):
                    exog = [exog]
                exog = self._arg('exog', exog, ListOfStrings)
                for each_exog in exog:
                    if each_exog not in cols:
                        msg = f"Please select exog from {cols}!"
                        logger.error(msg)
                        raise ValueError(msg)
            else:
                exog = cols

        data_ = data[[key] + [endog] + exog] if exog is not None else data[[key] + [endog]]
        #print('Fit data columns:\n', data_.columns)

        param_rows = [('FUNCTION', self.func, None, None)]
        for name in self.__pal_params:
            value, typ = self.__pal_params[name]
            if name == 'SEASON_START':
                param_rows.extend([('SEASON_START', element[0], element[1], None)
                                    for element in value])
            elif name == 'SEASONAL':
                param_rows.extend([('SEASONAL', value, None, None)])
            else:
                tpl = [_map_param(name, value, typ)]
                param_rows.extend(tpl)
        #print("param_rows:\n")
        #print(param_rows)

        self.fit_data = data_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'STATS', 'DECOMPOSED', 'PLACE_HOLDER1']
        outputs = ['#PAL_UNIFIED_TIMESERIES_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        model_tbl, stats_tbl, decompose_tbl, _ = outputs
        if not (check_pal_function_exist(conn, '%UNIFIED_TIMESERIES%', like=True) or \
            self._disable_hana_execution):
            msg = 'The version of SAP HANA does not support Unified Time Series!'
            logger.error(msg)
            raise ValueError(msg)
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_TIMESERIES',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise
        except Exception as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise

        #pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.statistics_ = conn.table(stats_tbl)
        self.decompose_ = conn.table(decompose_tbl)

        return self

    def make_future_dataframe(self, data=None, key=None, periods=1, increment_type='seconds'):
        """
        Create a new dataframe for time series prediction.

        Parameters
        ----------
        data : DataFrame, optional
            The training data contains the index.

            Defaults to the data used in the fit().

        key : str, optional
            The index defined in the training data.

            Defaults to the specified key in fit function or the data.index or the first column of the data.

        periods : int, optional
            The number of rows created in the predict dataframe.

            Defaults to 1.

        increment_type : {'seconds', 'days', 'months', 'years'}, optional
            The increment type of the time series.

            Defaults to 'seconds'.

        Returns
        -------
        DataFrame

        """
        if data is None:
            if hasattr(self, "training_data") and getattr(self, "training_data") is not None:
                data = self.training_data
            else:
                raise ValueError("data is required when make_future_dataframe is called before fit().")
        if key is None:
            if data.index is None:
                if hasattr(self, "key"):
                    if self.key is None:
                        key = data.columns[0]
                    else:
                        key = self.key
                else:
                    key = data.columns[0]
            else:
                key = data.index
        max_ = data.select(key).max()
        sec_max_ = data.select(key).distinct().sort_values(key, ascending=False).head(2).collect().iat[1, 0]
        delta = (max_ - sec_max_)
        is_int = 'INT' in data.get_table_structure()[key]
        if is_int:
            forecast_start, timedelta = max_ + delta, delta
        else:
            forecast_start, timedelta = max_ + delta, delta.total_seconds()
        timeframe = []
        if not is_int:
            if 'day' in increment_type.lower():
                increment_type = 'days'
                timedelta = round(timedelta / 86400)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one day.")
            elif 'month' in increment_type.lower():
                increment_type = 'months'
                timedelta = round(timedelta / 2592000)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one month.")
            elif 'year' in increment_type.lower():
                increment_type = 'years'
                timedelta = round(timedelta / 31536000)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one year.")
            else:
                increment_type = 'seconds'
        for period in range(0, periods):
            if is_int:
                timeframe.append("SELECT TO_INT({} + {} * {}) AS \"{}\" FROM DUMMY".format(forecast_start, timedelta, period, key))
            else:
                timeframe.append("SELECT ADD_{}('{}', {} * {}) AS \"{}\" FROM DUMMY".format(increment_type.upper(), forecast_start, timedelta, period, key))
        sql = ' UNION ALL '.join(timeframe)
        return data.connection_context.sql(sql).sort_values(key)

    def predict(self,
                data,
                key=None,
                exog=None,
                **kwargs):
        """
        Predict function.

        Parameters
        ----------
        data : DataFrame
            Data for prediction.

        key : str, optional
            Name of ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        exog : str or list of str, optional
            The column(s) of exogenous variables to be used in the model.

            Defaults to all non-ID columns if not provided.

        **kwargs : keyword arguments
            Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.


        Returns
        -------
        DataFrame 1
            DataFrame containing the forecast values and other related
            statistics(like standard error estimation, upper/lower quantiles).

        DataFrame 2
            DataFrame containing the trend/seasonal/regression components
            w.r.t. the forecast values.

        """
        # If func is SMOOTH, return statistics_ and decompose_ directly as PAL_UNIFIED_TIMESERIES_PREDICT is not supported for SMOOTH
        if self.func == 3:
            return self.statistics_, self.decompose_

        if getattr(self, 'model_') is None:
            raise FitIncompleteError()

        if data is None:
            raise ValueError('data is required for prediction!')

        # Get predict param mapping for current func
        predict_param_map = map_dict_predict.get(self.func_name, {})
        # Convert kwargs to PAL params using _params_check (same as fit)
        pal_predict_params = {}
        if kwargs:
            pal_predict_params = _params_check(input_dict=kwargs,
                                               param_map=predict_param_map,
                                               func=self.func_name)

        conn = data.connection_context
        index = data.index
        key = self._arg('key', key, str, not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                warn_msg = f"Discrepancy between the designated key column '{key}' " +\
                f"and the designated index column '{index}'."
                logger.warning(warn_msg)
        key = index if key is None else key
        if isinstance(exog, str):
            exog = [exog]
        exog = self._arg('exog', exog, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if self.func == 3:
            exog = None
        else:
            if exog is None:
                exog = cols
        data_ = data[[key] + exog] if exog is not None else data[[key]]
        #print('Predict data columns:\n', data_.columns)

        param_rows = []
        for name in pal_predict_params:
            value, typ = pal_predict_params[name]
            if name == 'ACCURACY_MEASURE':
                if isinstance(value, str):
                    value = [value]
                for each_ac in value:
                    param_rows.extend([('ACCURACY_MEASURE', None, None, each_ac)])
            else:
                tpl = [_map_param(name, value, typ)]
                param_rows.extend(tpl)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        out_tabs = ['FORECAST', 'DECOMPOSE', 'PLACE_HOLDER']
        out_tabs = [f'#PAL_UNIFIED_TIMESERIES_PREDICT_{tb}_TBL_{self.id}_{unique_id}' for tb in out_tabs]
        setattr(self, 'predict_data', data_)
        #print("param_rows for predict:\n", param_rows)
        if not (check_pal_function_exist(conn, '%UNIFIED_TIMESERIES_PREDICT%', like=True) or \
            self._disable_hana_execution):
            msg = 'The version of SAP HANA does not support Unified Time Series!'
            logger.error(msg)
            raise ValueError(msg)
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_TIMESERIES_PREDICT',
                                data_,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                *out_tabs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, out_tabs)
            raise
        except Exception as db_err:
            logger.exception(str(db_err))
            try_drop(conn, out_tabs)
            raise
        forecast = conn.table(out_tabs[0])
        decompose = conn.table(out_tabs[1])

        return forecast, decompose

class MassiveUnifiedTimeSeries(PALBase):
    """
    The Python wrapper for SAP HANA PAL Massive Unified Time Series function.
    The Massive Unified Time Series algorithms include:

    - Additive Model Time Series Analysis (AMTSA)
    - Auto Regressive Integrated Moving Average (ARIMA)
    - Bayesian Structural Time Series (BSTS)
    - Exponential Smoothing (SMOOTH)

    Parameters
    ----------
    func : str

        The name of a specified time series algorithm.

        The following algorithms are supported:

        - 'AMTSA': Additive Model Time Series Analysis
        - 'ARIMA': Auto Regressive Integrated Moving Average
        - 'BSTS': Bayesian Structural Time Series
        - 'SMOOTH': Auto Exponential Smoothing

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

        - **'AMTSA'** : :class:`~hana_ml.algorithms.pal.tsa.additive_model_time_series.AdditiveModelTimeSeriesAnalysis`
          AMTSA in UnifiedTimeSeries has some additional parameters, please see the following section.

          - target_type : str, optional
          - start_point : str, optional, specify a start point in type conversion
          - interval : int, optional, specify an interval in type conversion.
          - holiday : str, optional, add holiday to model in a json format, including name, timestamp, (optional) lower_window, and (optional) upper_window elements. For example: '{ "name": "New Year", "timestamp": "2025-01-01" }'

        - **'ARIMA'** : :class:`~hana_ml.algorithms.pal.tsa.auto_arima.AutoARIMA`

        - **'BSTS'** : :class:`~hana_ml.algorithms.pal.tsa.bsts.BSTS`

        - **'SMOOTH'** : :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.AutoExponentialSmoothing`

        For more parameter mappings of hana_ml and HANA PAL, please refer to the doc page: :ref:`param_mapping`

    Attributes
    ----------

    model_ : DataFrame
        Model information.

    statistics_ : DataFrame
        Statistics.

    decompose_ : DataFrame
        Decomposition values.

    error_msg_ : DataFrame
        Error message during the fit process.


    Examples
    --------

    >>> muts = MassiveUnifiedTimeSeries(func='AMTSA')

    Perform fit():

    >>> muts.fit(data=df, group_key='group_id', key="ID", endog='value', exog=["ex1", "ex2"])

    Attributes after fit:

    >>> muts.statistics_.collect()
    >>> muts.decompose_.collect()
    >>> muts.error_msg_.collect()

    Invoke predict():

    >>> forecast, decompose, error_msg = muts.predict(data=df_pred, group_key='group_id', key="ID", exog=["ex1", "ex2"])

    Output:

    >>> forecast.collect()
    >>> decompose.collect()
    >>> error_msg.collect()

    """
    def __init__(self,
                 func,
                 group_params=None,
                 **kwargs):
        super(MassiveUnifiedTimeSeries, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.func = self._arg('Function name', func, func_dict)
        self.func_name = func.upper()
        func_map = map_dict[self.func_name]
        self.special_group_name = 'PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID_RESETTABLE'

        self.params = {**kwargs}
        self.__pal_params = {}
        group_params = self._arg('group_params', group_params, dict)
        group_params = {} if group_params is None else group_params
        for group in group_params:
            self._arg('Parameters with group_key ' + str(group), group_params[group], dict)
        self.group_params = group_params
        for group in self.group_params:
            self.__pal_params[group] = {}
            self.__pal_params[group] = _params_check(input_dict=self.group_params[group],
                                                     param_map=func_map,
                                                     func=self.func_name)
        if self.params:
            self.__pal_params[self.special_group_name] = _params_check(input_dict=self.params,
                                                                        param_map=func_map,
                                                                        func=self.func_name)
        #print("self.__pal_params:\n", self.__pal_params)
        self.model_ = None
        self.statistics_ = None
        self.decompose_ = None
        self.fit_data = None
        self.error_msg_ = None

    @trace_sql
    def fit(self,
            data,
            group_key=None,
            key=None,
            endog=None,
            exog=None
            ):
        """
        Fit function.

        Parameters
        ----------
        data : DataFrame
            Training data.

        group_key : str, optional
            The column of group_key. Data type can be INT or NVARCHAR/VARCHAR.

            Defaults to the first column of data if the index columns of data is not provided.
            Otherwise, defaults to the first column of index columns.

        key : str, optional

            Name of ID column.

            Defaults to the first column of data if the index column of data is not provided
            and ``group_key`` column is eliminated. Otherwise, defaults to the second index column of data.

        endog : str, optional
            The column of time series to be fitted and predicted.

            Defaults to the first column of data after eliminating key and group_key columns.

        exog : str or list of str, optional
            The column(s) of exogenous regressors.

            If not specified, all columns except group_key, key and endog are treated as exogenous regressors.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        setattr(self, 'hanaml_fit_params', pal_param_register())
        setattr(self, "training_data", data)
        setattr(self, 'key', key)
        setattr(self, 'endog', endog)
        setattr(self, 'exog', None)

        cols = data.columns
        index = data.index
        group_key = self._arg('group_key', group_key, str)
        if index is not None:
            group_key = _col_index_check(group_key, 'group_key', index[0], cols)
        else:
            if group_key is None:
                group_key = cols[0]

        if group_key is not None and group_key not in cols:
            msg = "Please select group_key from {}!".format(cols)
            logger.error(msg)
            raise ValueError(msg)
        data_groups = list(data[[group_key]].collect()[group_key].drop_duplicates())
        param_keys = list(self.group_params.keys())
        if not self._disable_hana_execution:
            gid_type = data[[group_key]].dtypes()[0]
            if not all([(int(ky) if 'INT' in gid_type[1] else ky) in data_groups for ky in param_keys]):
                msg = 'Invalid group key identified in group parameters!'
                logger.error(msg)
                raise ValueError(msg)
        else:
            gid_type = {tp[0]:tp for tp in data.dtypes()}[group_key]
        cols.remove(group_key)

        key = self._arg('key', key, str)
        if index is not None:
            key = _col_index_check(key, 'key', index[1], cols)
        else:
            if key is None:
                key = cols[0]
        if key is not None and key not in cols:
            msg = f"Please select key from {cols}!"
            logger.error(msg)
            raise ValueError(msg)
        cols.remove(key)

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = f"Please select endog from {cols}!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]
        cols.remove(endog)

        if self.func == 3: # SMOOTH does not support exog
            exog = None
        else: # not SMOOTH
            if exog is not None:
                if isinstance(exog, str):
                    exog = [exog]
                exog = self._arg('exog', exog, ListOfStrings)
                for each_exog in exog:
                    if each_exog not in cols:
                        msg = f"Please select exog from {cols}!"
                        logger.error(msg)
                        raise ValueError(msg)
            else:
                exog = cols

        data_ = data[[group_key] + [key] + [endog] + exog] if exog is not None else data[[group_key] + [key] + [endog]]

        #print('Fit data columns:\n', data_.columns)
        param_rows = [(self.special_group_name, 'FUNCTION', self.func, None, None)] if hasattr(self, 'special_group_name') else []
        if self.group_params is not None:
            for group in self.__pal_params:
                param_rows.extend([(group, 'FUNCTION', self.func, None, None)])
                for name in self.__pal_params[group]:
                    value, typ = self.__pal_params[group][name]
                    if name == 'ACCURACY_MEASURE':
                        if isinstance(value, str):
                            value = [value]
                        for each_ac in value:
                            param_rows.extend([(group, 'ACCURACY_MEASURE', None, None, each_ac)])
                    elif name == 'SEASON_START':
                        for element in value:
                            param_rows.extend([(group, 'SEASON_START', element[0], element[1], None)])
                    elif name == 'SEASONAL':
                        param_rows.extend([(group, 'SEASONAL', value, None, None)])
                    else:
                        tpl = [tuple([group] + list(_map_param(name, value, typ)))]
                        param_rows.extend(tpl)
        #print("param_rows:\n", param_rows)

        self.fit_data = data_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'STATS', 'DECOMPOSED', 'ERROR', 'PLACE_HOLDER1']
        outputs = [f'#PAL_MASSIVE_UNIFIED_TIMESERIES_{tbl}_{self.id}_{unique_id}'
                   for tbl in outputs]
        model_tbl, stats_tbl, err_tbl, decompose_tbl, _ = outputs
        if not (check_pal_function_exist(conn, '%MASSIVE_TIMESERIES%', like=True) or \
            self._disable_hana_execution):
            msg = 'The version of SAP HANA does not support Massive Unified Time Series!'
            logger.error(msg)
            raise ValueError(msg)
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_MASSIVE_TIMESERIES',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise
        except Exception as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise

        self.model_ = conn.table(model_tbl)
        self.statistics_ = conn.table(stats_tbl)
        self.decompose_ = conn.table(decompose_tbl)
        self.error_msg_ = conn.table(err_tbl)

        return self

    def make_future_dataframe(self, data=None, key=None, group_key=None, periods=1, increment_type='seconds'):
        """
        Create a new dataframe for time series prediction.

        Parameters
        ----------
        data : DataFrame, optional
            The training data contains the index.

            Defaults to the data used in the fit().

        key : str, optional
            The index defined in the training data.

            Defaults to the specified key in fit() or the value in data.index or the PAL's default key column position.

        group_key : str, optional
            Specify the group id column.

            This parameter is only valid when ``massive`` is True.

            Defaults to the specified group_key in fit() or the first column of the dataframe.

        periods : int, optional
            The number of rows created in the predict dataframe.

            Defaults to 1.

        increment_type : {'seconds', 'days', 'months', 'years'}, optional
            The increment type of the time series.

            Defaults to 'seconds'.

        Returns
        -------
        DataFrame

        """
        if data is None:
            if hasattr(self, "training_data") and getattr(self, "training_data") is not None:
                data = self.training_data
            else:
                raise ValueError("data is required when make_future_dataframe is called before fit().")
        if group_key is None:
            if hasattr(self, "group_key"):
                if self.group_key is None:
                    group_key = data.columns[0]
                else:
                    group_key = self.group_key
            else:
                group_key = data.columns[0]
        if key is None:
            if data.index is None:
                if hasattr(self, "key"):
                    if self.key is None:
                        key = data.columns[1]
                    else:
                        key = self.key
                else:
                    key = data.columns[1]
            else:
                key = data.index
        group_id_type = data.get_table_structure()[group_key]
        group_list = data.select(group_key).distinct().collect()[group_key]
        timeframe = []
        for group in group_list:
            if 'INT' in group_id_type.upper():
                m_data = data.filter(f"{group_key}={group}")
            else:
                m_data = data.filter(f"{group_key}='{group}'")
            max_ = m_data.select(key).max()
            sec_max_ = m_data.select(key).distinct().sort_values(key, ascending=False).head(2).collect().iat[1, 0]
            delta = max_ - sec_max_
            is_int = 'INT' in m_data.get_table_structure()[key]
            if is_int:
                forecast_start, timedelta = max_ + delta, delta
            else:
                forecast_start, timedelta = max_ + delta, delta.total_seconds()

            if not is_int:
                if 'day' in increment_type.lower():
                    increment_type = 'days'
                    timedelta = round(timedelta / 86400)
                    if timedelta == 0:
                        raise ValueError("The interval between the training time series is less than one day.")
                elif 'month' in increment_type.lower():
                    increment_type = 'months'
                    timedelta = round(timedelta / 2592000)
                    if timedelta == 0:
                        raise ValueError("The interval between the training time series is less than one month.")
                elif 'year' in increment_type.lower():
                    increment_type = 'years'
                    timedelta = round(timedelta / 31536000)
                    if timedelta == 0:
                        raise ValueError("The interval between the training time series is less than one year.")
                else:
                    increment_type = 'seconds'

            inc_upper = increment_type.upper()
            for period in range(0, periods):
                if 'INT' in group_id_type.upper():
                    if is_int:
                        timeframe.append(f"SELECT {group} AS \"{group_key}\", TO_INT({forecast_start} + {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
                    else:
                        timeframe.append(f"SELECT {group} AS \"{group_key}\", ADD_{inc_upper}('{forecast_start}', {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
                else:
                    if is_int:
                        timeframe.append(f"SELECT '{group}' AS \"{group_key}\", TO_INT({forecast_start} + {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
                    else:
                        timeframe.append(f"SELECT '{group}' AS \"{group_key}\", ADD_{inc_upper}('{forecast_start}', {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
        sql = ' UNION ALL '.join(timeframe)

        return data.connection_context.sql(sql).sort_values([group_key, key])

    def predict(self,
                data,
                group_key=None,
                group_params=None,
                key=None,
                exog=None,
                **kwargs):
        """
        Predict function.

        Parameters
        ----------
        data : DataFrame
            Predict data.

        group_key : str, optional
            The column of group_key. Data type can be INT or NVARCHAR/VARCHAR.

            Defaults to the first column of data if the index columns of data is not provided.
            Otherwise, defaults to the first column of index columns.

        key : str, optional

            Name of ID column.

            Defaults to the first column of data if the index column of data is not provided
            and ``group_key`` column is eliminated. Otherwise, defaults to the second index column of data.

        exog : str or list of str, optional
            The column(s) of exogenous regressors.

            If not specified, all columns except key and endog are treated as exogenous regressors.

        group_params : dict, optional
            The input data for time series shall be divided into different
            groups with different time series parameters applied. This parameter specifies the parameter
            values of the chosen time series algorithm ``func`` w.r.t. different groups in a dict format,
            where keys corresponding to ``group_key`` while values should be a dict for time series algorithm
            parameter value assignments.
        """
        # If func is SMOOTH, return statistics_, decompose_ and error_msg_ directly as PAL_UNIFIED_MASSIVE_TIMESERIES_PREDICT is not supported for SMOOTH
        if self.func == 3:
            return self.statistics_, self.decompose_, self.error_msg_

        if data is None:
            raise ValueError('data is required for prediction!')

        if getattr(self, 'model_') is None:
            raise FitIncompleteError()

        predict_param_map = map_dict_predict.get(self.func_name, {})
        self.predict_params = {**kwargs}
        self.__predict_pal_params = {}
        group_params = self._arg('predict group_params', group_params, dict)
        group_params = {} if group_params is None else group_params
        for group in group_params:
            self._arg('Parameters with group_key ' + str(group), group_params[group], dict)
        self.predict_group_params = group_params
        for group in self.predict_group_params:
            self.__predict_pal_params[group] = {}
            self.__predict_pal_params[group] = _params_check(input_dict=self.predict_group_params[group],
                                                    param_map=predict_param_map,
                                                    func=self.func)
        if self.predict_params:
            self.__predict_pal_params[self.special_group_name] = _params_check(input_dict=self.predict_params,
                                                                               param_map=predict_param_map,
                                                                               func=self.func)
        #print("self.__predict_pal_params:\n", self.__predict_pal_params)
        conn = data.connection_context
        cols = data.columns
        index = data.index
        group_key = self._arg('group_key', group_key, str)
        if index is not None:
            group_key = _col_index_check(group_key, 'group_key', index[0], cols)
        else:
            if group_key is None:
                group_key = cols[0]

        if group_key is not None and group_key not in cols:
            msg = "Please select group_key from {}!".format(cols)
            logger.error(msg)
            raise ValueError(msg)
        data_groups = list(data[[group_key]].collect()[group_key].drop_duplicates())
        param_keys = list(self.group_params.keys())
        if not self._disable_hana_execution:
            gid_type = data[[group_key]].dtypes()[0]
            if not all([(int(ky) if 'INT' in gid_type[1] else ky) in data_groups for ky in param_keys]):
                msg = 'Invalid group key identified in group parameters!'
                logger.error(msg)
                raise ValueError(msg)
        else:
            gid_type = {tp[0]:tp for tp in data.dtypes()}[group_key]
        cols.remove(group_key)

        key = self._arg('key', key, str)
        if index is not None:
            key = _col_index_check(key, 'key', index[1], cols)
        else:
            if key is None:
                key = cols[0]
        if key is not None and key not in cols:
            msg = f"Please select key from {cols}!"
            logger.error(msg)
            raise ValueError(msg)
        cols.remove(key)

        if exog is not None: # self.func != 3
            if isinstance(exog, str):
                exog = [exog]
            exog = self._arg('exog', exog, ListOfStrings)
            for each_exog in exog:
                if each_exog not in cols:
                    raise ValueError(f"exog '{each_exog}' not in columns {cols}")
        else:
            exog = cols

        data_ = data[[group_key] + [key] + exog] if exog is not None else data[[group_key] + [key]]

        #print('Massive Predict data columns:\n', data_.columns)

        param_rows = []
        if self.predict_group_params is not None:
            for group in self.predict_group_params:
                for name in self.__predict_pal_params[group]:
                    value, typ = self.__predict_pal_params[group][name]
                    tpl = [tuple([group] + list(_map_param(name, value, typ)))]
                    param_rows.extend(tpl)
        #print("param_rows for predict:\n", param_rows)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        out_tabs = ['FORECAST', 'DECOMPOSE', 'ERROR', 'PLACE_HOLDER']
        out_tabs = [f'#PAL_UNIFIED_MASSIVE_TIMESERIES_PREDICT_{tb}_TBL_{self.id}_{unique_id}' for tb in out_tabs]
        setattr(self, 'predict_data', data_)

        if not (check_pal_function_exist(conn, '%MASSIVE_TIMESERIES_PREDICT%', like=True) or \
            self._disable_hana_execution):
            msg = 'The version of SAP HANA does not support Massive Unified Time Series Predict!'
            logger.error(msg)
            raise ValueError(msg)
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_MASSIVE_TIMESERIES_PREDICT',
                                data_,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                *out_tabs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, out_tabs)
            raise
        except Exception as db_err:
            logger.exception(str(db_err))
            try_drop(conn, out_tabs)
            raise
        forecast = conn.table(out_tabs[0])
        decompose = conn.table(out_tabs[1])
        error_msg = conn.table(out_tabs[2])

        return forecast, decompose, error_msg
