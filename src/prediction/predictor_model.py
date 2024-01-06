import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
from darts.models.forecasting.xgboost import XGBModel
from darts import TimeSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the XGBoost Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "XGBoost Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        lags: Union[int, List[int], Dict[str, Union[int, List[int]]], None] = None,
        lags_past_covariates: Union[
            int, List[int], Dict[str, Union[int, List[int]]], None
        ] = None,
        lags_future_covariates: Union[
            Tuple[int, int],
            List[int],
            Dict[str, Union[Tuple[int, int], List[int]]],
            None,
        ] = None,
        output_chunk_length: int = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        multi_models: Optional[bool] = True,
        use_exogenous: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new XGBoost Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = lags_past_covariates = forecast horizon * lags_forecast_ratio
                lags_future_covariates = (lags, forecast horizon)
                This parameters overides lags parameters.

            lags (Union[int, List[int], Dict[str, Union[int, List[int]]], None]):
                Lagged target series values used to predict the next time step/s.
                If an integer, must be > 0. Uses the last n=lags past lags; e.g. (-1, -2, …, -lags),
                where 0 corresponds the first predicted time step of each sample. If a list of integers, each value must be < 0.
                Uses only the specified values as lags.
                If a dictionary, the keys correspond to the series component names (of the first series when using multiple series) and the values correspond to the component lags (integer or list of integers).
                The key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some components are missing and the 'default_lags' key is not provided.

            lags_past_covariates (Union[int, List[int], Dict[str, Union[int, List[int]]], None]):
                Lagged past_covariates values used to predict the next time step/s. If an integer,
                must be > 0. Uses the last n=lags_past_covariates past lags; e.g. (-1, -2, …, -lags), where 0 corresponds to the first predicted time step of each sample.
                If a list of integers, each value must be < 0. Uses only the specified values as lags.
                If a dictionary, the keys correspond to the past_covariates component names (of the first series when using multiple series) and the values correspond to the component lags (integer or list of integers).
                The key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some components are missing and the 'default_lags' key is not provided.


            lags_future_covariates (Union[Tuple[int, int], List[int], Dict[str, Union[Tuple[int, int], List[int]]], None]):
                Lagged future_covariates values used to predict the next time step/s. If a tuple of (past, future), both values must be > 0.
                Uses the last n=past past lags and n=future future lags; e.g. (-past, -(past - 1), …, -1, 0, 1, …. future - 1), where 0 corresponds the first predicted time step of each sample.
                If a list of integers, uses only the specified values as lags. If a dictionary, the keys correspond to the future_covariates component names (of the first series when using multiple series) and the values correspond to the component lags (tuple or list of integers).
                The key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some components are missing and the 'default_lags' key is not provided.

            output_chunk_length (int):
              Number of time steps predicted at once (per chunk) by the internal model.
              It is not the same as forecast horizon n used in predict(), which is the desired number of prediction points generated using a one-shot- or auto-regressive forecast.
              Setting n <= output_chunk_length prevents auto-regression. This is useful when the covariates don't extend far enough into the future,
              or to prohibit the model from using future values of past and / or future covariates for prediction (depending on the model's covariate support).
              If not set, the forecast horizon will be used.

            n_estimators (int): The number of trees in the forest.

            max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

            multi_models (Optional[bool]):
                If True, a separate model will be trained for each future lag to predict.
                If False, a single model is trained to predict at step 'output_chunk_length' in the future. Default: True.

            use_exogenous (bool):
                Indicated if past covariates are used or not.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.data_schema = data_schema
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.multi_models = multi_models
        self.use_exogenous = use_exogenous
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )
        if lags_forecast_ratio:
            lags = self.data_schema.forecast_length * lags_forecast_ratio
            self.lags = lags

            if use_exogenous and self.data_schema.past_covariates:
                self.lags_past_covariates = lags

        if (
            use_exogenous
            and not lags_future_covariates
            and (
                self.data_schema.future_covariates
                or self.data_schema.time_col_dtype in ["DATE", "DATETIME"]
            )
        ):
            self.lags_future_covariates = list(range(0, data_schema.forecast_length))

        if not self.use_exogenous:
            self.lags_past_covariates = None
            self.lags_future_covariates = None

        if not self.output_chunk_length:
            self.output_chunk_length = self.data_schema.forecast_length

        self.model = XGBModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            multi_models=self.multi_models,
            **kwargs,
        )

    def _prepare_data(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
    ) -> Tuple[List, List, List]:
        """
        Puts the data into the expected shape by the forecaster.
        Drops the time column and puts all the target series as columns in the dataframe.

        Args:
            history (pd.DataFrame): The provided training data.
            data_schema (ForecastingSchema): The schema of the training data.


        Returns:
            Tuple[List, List, List]: Target, Past covariates and Future covariates.
        """
        targets = []
        past = []
        future = []

        future_covariates_names = data_schema.future_covariates
        if data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            date_col = pd.to_datetime(history[data_schema.time_col])
            year_col = date_col.dt.year
            month_col = date_col.dt.month
            year_col_name = f"{data_schema.time_col}_year"
            month_col_name = f"{data_schema.time_col}_month"
            history[year_col_name] = year_col
            history[month_col_name] = month_col
            future_covariates_names += [year_col_name, month_col_name]

            year_col = date_col.dt.year
            month_col = date_col.dt.month

        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.all_ids = all_ids
        scalers = {}
        for index, s in enumerate(all_series):
            if self.history_length:
                s = s.iloc[-self.history_length :]
            s.reset_index(inplace=True)

            past_scaler = MinMaxScaler()
            scaler = MinMaxScaler()
            s[data_schema.target] = scaler.fit_transform(
                s[data_schema.target].values.reshape(-1, 1)
            )

            scalers[index] = scaler
            static_covariates = None
            if self.use_exogenous and self.data_schema.static_covariates:
                static_covariates = s[self.data_schema.static_covariates]

            target = TimeSeries.from_dataframe(
                s,
                value_cols=data_schema.target,
                static_covariates=static_covariates.iloc[0]
                if static_covariates is not None
                else None,
            )

            targets.append(target)

            if data_schema.past_covariates:
                original_values = (
                    s[data_schema.past_covariates].values.reshape(-1, 1)
                    if len(data_schema.past_covariates) == 1
                    else s[data_schema.past_covariates].values
                )
                s[data_schema.past_covariates] = past_scaler.fit_transform(
                    original_values
                )
                past_covariates = TimeSeries.from_dataframe(
                    s[data_schema.past_covariates]
                )
                past.append(past_covariates)

        future_scalers = {}
        if future_covariates_names:
            for id, train_series in zip(all_ids, all_series):
                if self.history_length:
                    train_series = train_series.iloc[-self.history_length :]

                future_covariates = train_series[future_covariates_names]

                future_covariates.reset_index(inplace=True)
                future_scaler = MinMaxScaler()
                original_values = (
                    future_covariates[future_covariates_names].values.reshape(-1, 1)
                    if len(future_covariates_names) == 1
                    else future_covariates[future_covariates_names].values
                )
                future_covariates[
                    future_covariates_names
                ] = future_scaler.fit_transform(original_values)

                future_covariates = TimeSeries.from_dataframe(
                    future_covariates[future_covariates_names]
                )
                future_scalers[id] = future_scaler
                future.append(future_covariates)

        self.scalers = scalers
        self.future_scalers = future_scalers
        if not past or not self.use_exogenous:
            past = None
        if not future or not self.use_exogenous:
            future = None

        return targets, past, future

    def _prepare_test_data(
        self,
        data: pd.DataFrame,
    ) -> List:
        """
        Prepares testing data.

        Args:
            data (pd.DataFrame): Testing data.

        Returns (List): Training and testing future covariates concatenated together.

        """
        future = []
        data_schema = self.data_schema
        future_covariates_names = data_schema.future_covariates
        if data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            date_col = pd.to_datetime(data[data_schema.time_col])
            year_col = date_col.dt.year
            month_col = date_col.dt.month
            year_col_name = f"{data_schema.time_col}_year"
            month_col_name = f"{data_schema.time_col}_month"
            data[year_col_name] = year_col
            data[month_col_name] = month_col
            year_col = date_col.dt.year
            month_col = date_col.dt.month

        groups_by_ids = data.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        if future_covariates_names:
            for id, test_series in zip(all_ids, all_series):
                future_covariates = test_series[future_covariates_names]

                future_covariates.reset_index(inplace=True)
                future_scaler = self.future_scalers[id]
                original_values = (
                    future_covariates[future_covariates_names].values.reshape(-1, 1)
                    if len(future_covariates_names) == 1
                    else future_covariates[future_covariates_names].values
                )

                future_covariates[future_covariates_names] = future_scaler.transform(
                    original_values
                )

                future_covariates = TimeSeries.from_dataframe(
                    future_covariates[future_covariates_names]
                )
                future.append(future_covariates)

        if not future or not self.use_exogenous:
            future = None
        else:
            for index, (train_covariates, test_covariates) in enumerate(
                zip(self.training_future_covariates, future)
            ):
                train_values = train_covariates.values()
                test_values = test_covariates.values()

                full_values = np.concatenate((train_values, test_values), axis=0)
                full_series = TimeSeries.from_values(full_values)

                future[index] = full_series

        return future

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate LinearRegression model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.

        """
        np.random.seed(self.random_state)
        targets, past_covariates, future_covariates = self._prepare_data(
            history=history,
            data_schema=data_schema,
        )

        self.model.fit(
            targets,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        self._is_trained = True
        self.data_schema = data_schema
        self.targets_series = targets
        self.past_covariates = past_covariates
        self.training_future_covariates = future_covariates

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        future_covariates = self._prepare_test_data(test_data)

        predictions = self.model.predict(
            n=self.data_schema.forecast_length,
            series=self.targets_series,
            past_covariates=self.past_covariates,
            future_covariates=future_covariates,
        )
        prediction_values = []
        for index, prediction in enumerate(predictions):
            prediction = prediction.pd_dataframe()
            values = prediction.values
            values = self.scalers[index].inverse_transform(values)
            prediction_values += list(values)

        test_data[prediction_col_name] = np.array(prediction_values)
        return test_data

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
        data_schema=data_schema,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
