import logging
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.base import BaseEstimator, is_regressor
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSet(Enum):
    US26 = "us26"
    EURO26 = "euro28"


class Statistics(Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"


class Metrics(Enum):
    R2 = "R^2 score"
    MSE = "MSE score"
    MAE = "MAE score"


def load_and_process_X(dataset_name: DataSet, num_files: int, statistics: List[Statistics]) -> pd.DataFrame:
    df_combined = pd.DataFrame()

    for j in range(num_files):
        df = pd.read_csv(f"data/demands/{dataset_name.value}/demands-{j}.csv", sep=";")
        df_modified = df.drop(columns=['source', 'destination'], axis=1)

        for i in range(100, 651, 25):
            one_row = pd.DataFrame({'nrOfRequests': i}, index=[0])

            if Statistics.SUM in statistics:
                sum_values = df_modified.iloc[:i].sum().rename(lambda x: f"sum {x}").to_frame().T
                one_row = pd.concat([one_row, sum_values], axis=1)

            if Statistics.MIN in statistics:
                min_values = df_modified.iloc[:i].min().rename(lambda x: f"min {x}").to_frame().T
                one_row = pd.concat([one_row, min_values], axis=1)

            if Statistics.MAX in statistics:
                max_values = df_modified.iloc[:i].max().rename(lambda x: f"max {x}").to_frame().T
                one_row = pd.concat([one_row, max_values], axis=1)

            if Statistics.MEAN in statistics:
                mean_values = df_modified.iloc[:i].mean().rename(lambda x: f"mean {x}").to_frame().T
                one_row = pd.concat([one_row, mean_values], axis=1)

            df_combined = pd.concat([df_combined, one_row], ignore_index=True)

    logging.info(f'Finished loading {dataset_name} with {num_files} files and {len(statistics)} statistics')
    return df_combined


def load_and_process_y(dataset: DataSet, num_files: int) -> pd.DataFrame:
    df_y_combined = pd.DataFrame()

    for j in range(num_files):
        df = pd.read_csv(f"data/demands/{dataset.value}/active_transceivers-{j}.csv", sep=";")
        df_modified = df.drop(columns=['nrOfRequests'], axis=1)
        df_y_combined = pd.concat([df_y_combined, df_modified], ignore_index=True)

    logging.info(f'Finished loading {dataset} with {num_files} files')
    return df_y_combined


def evaluate_models(models: List[BaseEstimator], X: pd.DataFrame, y: pd.DataFrame,
                    random_state: int, use_pca: bool = False,
                    n_components: Optional[float] = None,
                    metric: Optional[List[Metrics]] = None) -> pd.DataFrame:
    metrics_scores = []
    rskf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=random_state)

    for model in models:
        if not is_regressor(model):
            raise ValueError(f"The model {model} is not a regressor.")

        pipeline_steps = [StandardScaler()]
        if use_pca and n_components:
            pipeline_steps.append(PCA(n_components=n_components))
        pipeline_steps.append(model)
        pipeline = make_pipeline(*pipeline_steps)

        model_name = type(model).__name__
        if hasattr(model, 'n_neighbors'):
            model_name += f" (n_neighbors={model.n_neighbors})"

        all_fold_metrics = {Metrics.R2.value: [], Metrics.MSE.value: [], Metrics.MAE.value: []}
        for train_idx, test_idx in rskf.split(X, y):
            clf = clone(pipeline)
            clf.fit(X.iloc[train_idx], y[train_idx])
            y_pred = clf.predict(X.iloc[test_idx])

            if metric is None or Metrics.R2 in metric:
                all_fold_metrics[Metrics.R2.value].append(r2_score(y[test_idx], y_pred))
            if metric is None or Metrics.MSE in metric:
                all_fold_metrics[Metrics.MSE.value].append(mean_squared_error(y[test_idx], y_pred))
            if metric is None or Metrics.MAE in metric:
                all_fold_metrics[Metrics.MAE.value].append(mean_absolute_error(y[test_idx], y_pred))

        metrics_scores.append({
            'Model': model_name,
            Metrics.R2.value: np.mean(all_fold_metrics[Metrics.R2.value]),
            Metrics.MSE.value: np.mean(all_fold_metrics[Metrics.MSE.value]),
            Metrics.MAE.value: np.mean(all_fold_metrics[Metrics.MAE.value])
        })

        logging.info(f'Finished evaluating {model_name}')

    return pd.DataFrame(metrics_scores)


def plot_metrics_by_dataset_size(dataset_name: DataSet, models: List[Union[BaseEstimator, Pipeline]],
                                 num_files: int, statistics: List[Statistics],
                                 random_state: int, use_pca: bool,
                                 n_components: Optional[float], metrics: Optional[List[Metrics]] = None):
    if metrics is None:
        metrics = [Metrics.R2, Metrics.MSE, Metrics.MAE]

    all_model_metrics = []
    num_features = 0

    for file_index in range(num_files):
        logging.info(f"Processing the {file_index + 1} dataset")
        X = load_and_process_X(dataset_name, file_index + 1, statistics)
        y = load_and_process_y(dataset_name, file_index + 1)
        num_features = X.shape[1]

        model_metrics: pd.DataFrame = evaluate_models(models, X, y.values.ravel(), random_state, use_pca, n_components, metrics)

        model_metrics['Dataset Size'] = file_index + 1
        all_model_metrics.append(model_metrics)

    combined_metrics_df = pd.concat(all_model_metrics)

    for metric in metrics:
        title = f'Model Performance by Dataset Size - {metric.name} (Base features: {num_features})'
        if use_pca and n_components:
            title += f' (with PCA of {n_components} components)'
        plt.figure(figsize=(10, 6))
        for model_name in combined_metrics_df['Model'].unique():
            subset = combined_metrics_df[combined_metrics_df['Model'] == model_name]
            plt.plot(subset['Dataset Size'], subset[metric.value], label=model_name)
        plt.title(title)
        plt.xlabel('Number of Dataset Files')
        plt.ylabel(metric.value)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # df_X = load_and_process_X(DataSet.EURO26, 100, [Statistics.MIN, Statistics.MAX, Statistics.MEAN])
    # df_y = load_and_process_y(DataSet.EURO26, 100)
    # print(df_X)
    # print(df_y)
    # pick_best_model(df_X, df_y)
    random_state = 41
    base_models = [
        KNeighborsRegressor(n_neighbors=3),
        # KNeighborsRegressor(n_neighbors=7),
        SVR(),
        # DecisionTreeRegressor(random_state=random_state),
        # RandomForestRegressor(random_state=random_state),
        GradientBoostingRegressor(random_state=random_state),
        # AdaBoostRegressor(random_state=random_state),
        # Lasso(),
        Ridge(),
        # ElasticNet(),
    ]
    plot_metrics_by_dataset_size(DataSet.EURO26, base_models, 2, [Statistics.SUM, Statistics.MIN], 42, True, 0.90,
                                 None)
