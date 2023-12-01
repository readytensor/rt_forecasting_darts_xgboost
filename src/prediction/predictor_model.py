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
        lags: Union[int, List[int], Dict[str, Union[int, List[int]]], None] = None,
        lags_past_covariates: Union[
            int, List[int], Dict[str, Union[int, List[int]]], None
        ] = None,
        lags_future_covariates: Union[
            Tuple[int, int],
            List[int],
            Dict[str, Union[Tuple[int, int], List[int]]],
            None,
        ] = (5, 1),
        output_chunk_length: int = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        multi_models: Optional[bool] = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new XGBoost Forecaster

        Args:

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

            n_estimators (int): The number of trees in the forest.

            max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

            multi_models (Optional[bool]):
                If True, a separate model will be trained for each future lag to predict.
                If False, a single model is trained to predict at step 'output_chunk_length' in the future. Default: True.

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
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs

        if not data_schema.past_covariates:
            self.lags_past_covariates = None

        if not data_schema.future_covariates:
            self.lags_future_covariates = None

        self.history_length = None
        if kwargs.get("history_length"):
            self.history_length = kwargs["history_length"]
            kwargs.pop("history_length")

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
        history_length: int = None,
        test_dataframe: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Puts the data into the expected shape by the forecaster.
        Drops the time column and puts all the target series as columns in the dataframe.

        Args:
            history (pd.DataFrame): The provided training data.
            data_schema (ForecastingSchema): The schema of the training data.

        Returns:
            pd.DataFrame: The processed data.
        """
        targets = []
        past = []
        future = []

        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.all_ids = all_ids

        for s in all_series:
            if history_length:
                s = s.iloc[-self.history_length :]
            s.reset_index(inplace=True)
            target = TimeSeries.from_dataframe(s, value_cols=data_schema.target)
            targets.append(target)

            if data_schema.past_covariates:
                past_covariates = TimeSeries.from_dataframe(
                    s[data_schema.past_covariates]
                )
                past.append(past_covariates)

        if data_schema.future_covariates:
            test_groups_by_ids = test_dataframe.groupby(data_schema.id_col)
            test_all_series = [
                test_groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
                for id_ in all_ids
            ]

            for train_series, test_series in zip(all_series, test_all_series):
                if history_length:
                    train_series = train_series.iloc[-self.history_length :]
                    test_series = test_series.iloc[-self.history_length :]

                train_future_covariates = train_series[data_schema.future_covariates]
                test_future_covariates = test_series[data_schema.future_covariates]
                future_covariates = pd.concat(
                    [train_future_covariates, test_future_covariates], axis=0
                )
                future_covariates.reset_index(inplace=True)
                future_covariates = TimeSeries.from_dataframe(future_covariates)
                future.append(future_covariates)

        if not past:
            past = None
        if not future:
            future = None
        return targets, past, future

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
        test_dataframe: pd.DataFrame = None,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate XGBoost model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
            test_dataframe (pd.DataFrame): The testing data (needed only if the data contains future covariates).
        """
        np.random.seed(self.random_state)
        targets, past_covariates, future_covariates = self._prepare_data(
            history=history,
            history_length=history_length,
            data_schema=data_schema,
            test_dataframe=test_dataframe,
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
        self.future_covariates = future_covariates

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

        predictions = self.model.predict(
            n=self.data_schema.forecast_length,
            series=self.targets_series,
            past_covariates=self.past_covariates,
            future_covariates=self.future_covariates,
        )
        prediction_values = []
        for prediction in predictions:
            prediction = prediction.pd_dataframe()
            values = prediction.values
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
    testing_dataframe: pd.DataFrame = None,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.
        test_dataframe (pd.DataFrame): The testing data (needed only if the data contains future covariates).

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
        history_length=model.history_length,
        test_dataframe=testing_dataframe,
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
