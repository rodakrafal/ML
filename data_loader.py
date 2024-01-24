import csv
import logging
import os
import re
from enum import Enum
from typing import List


class DataSet(Enum):
    US26 = "us26"
    EURO26 = "euro28"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
array_of_names = [f'{x}' for x in range(5, 1445, 5)]


def is_txt_file(filename: str) -> bool:
    return filename.endswith('.txt')


def extract_last_number(filename: str) -> int:
    match = re.search(r'\d+$', filename)
    return int(match.group()) if match else None


def process_txt_file(writer, full_path: str, index_to_cut: List[int]) -> None:
    with open(full_path, 'r') as file:
        stripped = [line.strip() for line in file]
        filtered_lines = [item for i, item in enumerate(stripped) if i not in index_to_cut]
        writer.writerow(filtered_lines)


def load_demand(directory: str) -> None:
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return

    path_split = directory.split(os.sep)
    number = extract_last_number(path_split[-1])
    index_to_cut = [0, 3]
    csv_file_name = f'demands-{number}.csv'
    csv_file_path = os.path.join(path_split[0], "demands", path_split[1], csv_file_name)

    with open(csv_file_path, 'w', encoding='UTF8', newline='') as out_file:
        writer = csv.writer(out_file, delimiter=';')
        writer.writerow(['source', 'destination', *array_of_names])

        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for filename in filter(is_txt_file, os.listdir(subdir_path)):
                    full_path = os.path.join(subdir_path, filename)
                    if number > 9:
                        index_to_cut = [2]
                    process_txt_file(writer, full_path, index_to_cut)

    logging.info(f'Finished creating {csv_file_name} for {directory} in {csv_file_path}')


def load_demands(*datasets: DataSet) -> None:
    for dataset in datasets:
        root_directory = os.path.join("data", dataset.value)
        for directory in os.listdir(root_directory):
            load_demand(os.path.join(root_directory, directory))

    logging.info(f'Finished loading {len(datasets)} datasets')


def load_transceiver(directory: str) -> None:
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return

    path_split = directory.split(os.sep)
    number = extract_last_number(path_split[-1])
    csv_file_name = f'active_transceivers-{number}.csv'
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

    logging.info(f'Finished creating {csv_file_name} for {directory} in {csv_file_path}')


def load_transceivers(*datasets: DataSet) -> None:
    for dataset in datasets:
        root_directory = os.path.join("data", dataset.value)
        for directory in os.listdir(root_directory):
            load_transceiver(os.path.join(root_directory, directory))


if __name__ == '__main__':
    load_demands(DataSet.EURO26, DataSet.US26)
    # load_transceivers(DataSet.EURO26, DataSet.US26)
