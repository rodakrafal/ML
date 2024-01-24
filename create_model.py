import logging
import os
from enum import Enum
from io import StringIO
from random import random
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.evaluate import combined_ftest_5x2cv
from sklearn import clone
from sklearn.base import BaseEstimator, is_regressor
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

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


def load_and_process_random_X_y(dataset: DataSet, num_files: int, statistics: List[Statistics]) -> [pd.DataFrame, pd.DataFrame]:
    list_of_index = random.sample(range(100), num_files)
    df_y_combined = pd.DataFrame()
    df_X_combined = pd.DataFrame()

    for j in list_of_index:

        df = pd.read_csv(f"data/demands/{dataset.value}/active_transceivers-{j}.csv", sep=";")
        df_modified = df.drop(columns=['nrOfRequests'], axis=1)
        df_y_combined = pd.concat([df_y_combined, df_modified], ignore_index=True)

        df = pd.read_csv(f"data/demands/{dataset.value}/demands-{j}.csv", sep=";")
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

            df_X_combined = pd.concat([df_modified, one_row], ignore_index=True)

    logging.info(f'Finished loading {dataset} with {num_files} files and {len(statistics)} statistics')

    return df_X_combined, df_y_combined

def get_model_name(model: BaseEstimator) -> str:
    model_name = type(model).__name__
    if hasattr(model, 'n_neighbors'):
        model_name += f" (n_neighbors={model.n_neighbors})"
    return model_name


def evaluate_models(models: List[BaseEstimator], X: pd.DataFrame, y: pd.DataFrame, metrics: List[Metrics],
                    random_state: int, use_pca: bool = False,
                    n_components: Optional[float] = None) -> pd.DataFrame:
    n_splits = 2
    n_repeats = 5
    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    metrics_scores = [np.zeros(shape=(len(models), n_repeats * n_splits)) for _ in range(len(metrics))]
    number_of_all_iter = n_repeats * n_splits * len(models)

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_idx = int(fold_idx / n_splits)
        for model in models:
            if not is_regressor(model):
                raise ValueError(f"The model {model} is not a regressor.")

            pipeline_steps = [StandardScaler()]
            if use_pca and n_components:
                pipeline_steps.append(PCA(n_components=n_components))
            pipeline_steps.append(model)
            pipeline = make_pipeline(*pipeline_steps)

            model_name = get_model_name(model)
            clf = clone(pipeline)
            clf.fit(X.iloc[train_idx], y[train_idx])
            y_pred = clf.predict(X.iloc[test_idx])

            split_idx = fold_idx % n_splits

            for metric_idx, metric in enumerate(metrics):
                if metric == Metrics.R2:
                    score = r2_score(y[test_idx], y_pred)
                elif metric == Metrics.MSE:
                    score = mean_squared_error(y[test_idx], y_pred)
                elif metric == Metrics.MAE:
                    score = mean_absolute_error(y[test_idx], y_pred)
                else:
                    raise ValueError(f"Unknown metric {metric}")

                metrics_scores[metric_idx][models.index(model), fold_idx] = score

            left_number_of_iter = number_of_all_iter - (fold_idx * len(models) + models.index(model) + 1)

            logging.info(
                f'Finished evaluating {model_name} on fold {fold_idx}, repeat {repeat_idx} and split {split_idx} left: {left_number_of_iter}')

    logging.info("Finished evaluating all models")
    metric_names = [metric.value for metric in metrics]
    average_scores = [scores.mean(axis=1) for scores in metrics_scores]
    model_names = [get_model_name(model) for model in models]
    averaged_metrics_df = pd.DataFrame(average_scores, index=metric_names,
                                       columns=model_names).T

    return averaged_metrics_df


def evaluate_model_with_diff_examples(dataset_name: DataSet, model: BaseEstimator, num_files: int, statistics: List[Statistics],
                   random_state: int, n_components: int | float = None,
                   metrics: Optional[List[Metrics]] = None):

    if metrics is None:
        metrics = [Metrics.R2, Metrics.MSE, Metrics.MAE]

    all_metrics = []

    for use_pca in [True, False]:
        pca_label = "with PCA" if use_pca else "without PCA"
        for x in range(10):
            for file_index in range(0, num_files + 1, 25):
                if file_index == 0:
                    file_index = 1
                X, y = load_and_process_random_X_y(dataset_name, file_index, statistics)

                model_metrics: pd.DataFrame = evaluate_models([model], X, y.values.ravel(), metrics, random_state, use_pca,
                                                              n_components)

                model_metrics['Model'] = get_model_name(model)
                model_metrics['PCA'] = pca_label
                model_metrics['Num Request sets'] = file_index
                model_metrics['Num Features'] = X.shape[1]
                model_metrics['Num Samples'] = len(y)

                all_metrics.append(model_metrics)

    combined_metrics_df = pd.concat(all_metrics, ignore_index=True)

    combined_metrics_df = combined_metrics_df[
        ['Model', 'PCA', 'Num Request sets', 'Num Features', 'Num Samples'] + [metric.value for metric in metrics]]
    combined_metrics_df.reset_index(drop=True, inplace=True)

    return combined_metrics_df


def evaluate_model_with_diff_features(dataset_name: DataSet, model: BaseEstimator,
                   random_state: int, n_components: int | float = None,
                   metrics: Optional[List[Metrics]] = None):

    if metrics is None:
        metrics = [Metrics.R2, Metrics.MSE, Metrics.MAE]

    all_metrics = []

    for use_pca in [True, False]:
        pca_label = "with PCA" if use_pca else "without PCA"
        for statistic in [[Statistics.SUM], [Statistics.SUM, Statistics.MAX],
                          [Statistics.SUM, Statistics.MAX, Statistics.MEAN]]:
            X = load_and_process_X(dataset_name, 100, statistic)
            y = load_and_process_y(dataset_name, 100)

            model_metrics: pd.DataFrame = evaluate_models([model], X, y.values.ravel(), metrics, random_state, use_pca,
                                                          n_components)

            model_metrics['Model'] = get_model_name(model)
            model_metrics['PCA'] = pca_label
            model_metrics['Num Request sets'] = 100
            model_metrics['Num Features'] = X.shape[1]
            model_metrics['Num Samples'] = len(y)

            all_metrics.append(model_metrics)

    combined_metrics_df = pd.concat(all_metrics, ignore_index=True)

    combined_metrics_df = combined_metrics_df[
        ['Model', 'PCA', 'Num Request sets', 'Num Features', 'Num Samples'] + [metric.value for metric in metrics]]
    combined_metrics_df.reset_index(drop=True, inplace=True)

    return combined_metrics_df


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

        model_metrics: pd.DataFrame = evaluate_models(models, X, y.values.ravel(), random_state, use_pca, n_components,
                                                      metrics)

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


def compare_models_with_ttest(models: List[BaseEstimator], scoring: Metrics) -> pd.DataFrame:
    model_names = [get_model_name(model) for model in models]
    models_combinations = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    for dataset in [DataSet.EURO26, DataSet.US26]:
        df_X = load_and_process_X(dataset, 100, [Statistics.SUM])
        df_y = load_and_process_y(dataset, 100)
        for metric in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
            p_value_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
            t_statistic_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
            for model1, model2 in models_combinations:
                t_val, p_val = combined_ftest_5x2cv(estimator1=model1,
                                                    estimator2=model2,
                                                    X=df_X, y=df_y.values.ravel(),
                                                    scoring=metric,
                                                    random_seed=1)
                p_value_matrix.loc[get_model_name(model2), get_model_name(model1)] = p_val
                t_statistic_matrix.loc[get_model_name(model2), get_model_name(model1)] = t_val

                print(
                    f"Model {get_model_name(model1)} and {get_model_name(model2)} are significantly different for {scoring.value} with p-value {p_val} and t-statistic {t_val}")

            p_value_matrix.to_csv(f"results/{dataset}_{metric}_p_value.csv")
            t_statistic_matrix.to_csv(f"results/{dataset}_{metric}_t_statistic.csv")
    return p_value_matrix


def format_csv(alpha=0.05):
    model_name_map = {
        "KNeighborsRegressor (n_neighbors=3)": "KNN3",
        "KNeighborsRegressor (n_neighbors=7)": "KNN7",
        "DecisionTreeRegressor": "DTR",
        "RandomForestRegressor": "RFR",
        "GradientBoostingRegressor": "GBR",
        "AdaBoostRegressor": "ADR",
        "ElasticNet": "EN",
    }

    def map_model_names(name):
        return model_name_map.get(name, name)

    def format_p_value(p):
        if pd.isna(p):
            return ""
        elif p >= alpha:
            return f"{p:.4f}*"
        else:
            return f"{p:.4f}"

    cwd = os.getcwd()
    directory = f"{cwd}/results"
    for file in os.listdir(directory):
        csv_path = f"{directory}/{file}"
        csv_data = pd.read_csv(csv_path).to_csv(index=False)
        csv_output_path = f"{cwd}/formatted_results/{file}"
        df = pd.read_csv(StringIO(csv_data))
        df.set_index(df.iloc[:, 0].values, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)

        formatted_csv_df_renamed = df.map(lambda x: format_p_value(x) if isinstance(x, float) else x)
        formatted_csv_df_renamed.columns = [map_model_names(name) for name in formatted_csv_df_renamed.columns]
        formatted_csv_df_renamed.index = [map_model_names(name) for name in formatted_csv_df_renamed.index]

        header = formatted_csv_df_renamed.columns
        formatted_csv_renamed = formatted_csv_df_renamed.to_csv(header=False)

        with open(csv_output_path, 'w') as file:
            file.write(formatted_csv_renamed)
            file.write(',' + ','.join(header) + '\n')


if __name__ == '__main__':
    random_state = 41
    # format_csv()
    # df_X = load_and_process_X(DataSet.EURO26, 100, [Statistics.SUM])
    # df_y = load_and_process_y(DataSet.EURO26, 100)
    # # print(df_X)
    #
    # # print(df_y)
    # # pick_best_model(df_X, df_y)
    # rskf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)
    # #
    # base_models = [
    #     # KNeighborsRegressor(n_neighbors=3),
    #     # KNeighborsRegressor(n_neighbors=7),
    #     # SVR(),
    #     # DecisionTreeRegressor(random_state=random_state),
    #     # RandomForestRegressor(random_state=random_state),
    #     GradientBoostingRegressor(random_state=random_state),
    #     # AdaBoostRegressor(random_state=random_state),
    #     # Lasso(),
    #     # Ridge(),
    #     # ElasticNet(),
    # ]
    # result = evaluate_models(base_models, df_X, df_y.values.ravel(), [Metrics.R2, Metrics.MSE, Metrics.MAE],
    #                          random_state, False, None)
    # print(result)
    # metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']  # The metrics to test
    # friedman_test_results = evaluate_and_compare_models(base_models, df_X, df_y, rskf, metrics)
    # print(friedman_test_results)
    # ttest_test_results = compare_models_with_ttest(base_models, Metrics.R2)
    # print(ttest_test_results)
    # cwd = os.getcwd()
    # csv_data = pd.read_csv('results/DataSet.US26_r2_p_value.csv').to_csv(index=False)
    # latex_file_path = f'{cwd}/formatted_results/DataSet.US26_r2_p_value_table.csv'
    # plot_metrics_by_dataset_size(DataSet.EURO26, base_models, 2, [Statistics.SUM, Statistics.MIN], 42, True, 0.90,
    #                              None)
    # t_test_models(DataSet.EURO26, base_models, 2, [Statistics.SUM, Statistics.MIN], 42, True, 0.90, None)

    # random_state = 41
    # dataset_name = DataSet.EURO26
    # num_files = 100
    # models = [GradientBoostingRegressor(random_state=random_state)]
    # evaluate_model(dataset_name, GradientBoostingRegressor(random_state=random_state), num_files, [Statistics.SUM], random_state, 0.90, None)

    results_euro = evaluate_model_with_diff_features(DataSet.EURO26, RandomForestRegressor(random_state=random_state), random_state, 0.95, None)
    print("Metrics for the EURO 28 topology:")
    print(results_euro)
    # results_us = evaluate_model(DataSet.US26, RandomForestRegressor(random_state=random_state), num_files,
    #                             [Statistics.SUM, Statistics.MAX], random_state, 0.95, None)

    # Display the metrics
    # print("\nMetrics for the US 26 topology:")
    # print(results_us)
