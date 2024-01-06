import csv
import logging
import os
from enum import Enum

import pandas as pd

class DataSet(Enum):
    US26 = "us26"
    EURO26 = "euro28"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
array_of_names = [f'{x}' for x in range(5, 1445, 5)]


def is_txt_file(filename: str) -> bool:
    return filename.endswith('.txt')


def process_txt_file(writer, full_path: str) -> None:
    with open(full_path, 'r') as file:
        stripped = [line.strip() for line in file]
        filtered_lines = [item for i, item in enumerate(stripped) if i not in [0, 3]]
        writer.writerow(filtered_lines)


def load_demand(directory: str) -> None:
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return

    path_split = directory.split(os.sep)
    csv_file_name = f'demands-{directory[-1]}.csv'
    csv_file_path = os.path.join(path_split[0], "demands", path_split[1], csv_file_name)

    with open(csv_file_path, 'w', encoding='UTF8', newline='') as out_file:
        writer = csv.writer(out_file, delimiter=';')
        writer.writerow(['source', 'destination', *array_of_names])

        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for filename in filter(is_txt_file, os.listdir(subdir_path)):
                    full_path = os.path.join(subdir_path, filename)
                    process_txt_file(writer, full_path)

    logging.info(f'Finished creating {csv_file_name} for {directory}')


def load_demands(*datasets: DataSet) -> None:
    for dataset in datasets:
        root_directory = os.path.join("data", dataset.value)
        for directory in os.listdir(root_directory):
            load_demand(os.path.join(root_directory, directory))

def load_transceiver(directory: str) -> None:
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return

    path_split = directory.split(os.sep)
    csv_file_name = f'active_transceivers-{directory[-1]}.csv'
    csv_file_path = os.path.join(path_split[0], "demands", path_split[1], csv_file_name)

    with open(csv_file_path, 'w', encoding='UTF8', newline='') as out_file:
        writer = csv.writer(out_file, delimiter=';')
        writer.writerow(['nrOfRequests', 'target'])

        for subdir in os.listdir(directory):
            file_path = os.path.join(directory, subdir)
            if not os.path.isdir(file_path) and "active_transceivers" in subdir:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.strip() and not line.startswith('nrOfRequests'):
                            nrOfRequests, target = line.strip().split()
                            writer.writerow([nrOfRequests, target])


    logging.info(f'Finished creating {csv_file_name} for {directory}')

def load_transceivers(*datasets: DataSet) -> None:
    for dataset in datasets:
        root_directory = os.path.join("data", dataset.value)
        for directory in os.listdir(root_directory):
            load_transceiver(os.path.join(root_directory, directory))


def create_dataframe() -> None:
    df_combined = pd.DataFrame()
    for j in range(0, 10):
        df = pd.read_csv(f"data/demands/euro28/demands-{j}.csv", sep=";")
        df_modified = df.copy()
        df_modified = df_modified.drop(columns=['source', 'destination'], axis=1)
        tmp = pd.DataFrame()
        for i in range(100, 651, 25):
            x = df_modified.iloc[:i][:].min()
            x = x.to_frame().T
            x.rename(columns={old_name: f"min {old_name}" for old_name in x.columns}, inplace=True)
            y = df_modified.iloc[:i][:].max()
            y = y.to_frame().T
            y.rename(columns={old_name: f"max {old_name}" for old_name in y.columns}, inplace=True)
            z = df_modified.iloc[:i][:].mean()
            z = z.to_frame().T
            z.rename(columns={old_name: f"mean {old_name}" for old_name in z.columns}, inplace=True)
            one_row = pd.concat([z, y, x], axis=1)
            one_row['nrOfRequests'] = i
            tmp = pd.concat([tmp, one_row], ignore_index=True)

        last_column = tmp.columns[-1]
        new_columns = [last_column] + list(tmp.columns[:-1])
        tmp = tmp[new_columns]
        df_combined = pd.concat([df_combined, tmp], ignore_index=True)
    print(df_combined)


if __name__ == '__main__':
    # load_demands(DataSet.EURO26, DataSet.US26)
    create_dataframe()
