import logging
import os
import os.path
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Union, Literal, NamedTuple
from src.data.loader import load_young_old_labels, load_tsv_files
from src.feature_selection import select_features_from_paper
from src.data.preprocessor import full_preprocessing_y_o_labels, preprocess_data, preprocess_huadong
from src.utils import Config



FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'


def load_preprocessed_data(fudan_filepath, huadong_filepath_1, huadong_filepath_2, labels_path, metadata_path, group, select_features):
    data = load_tsv_files(fudan_filepath)
    huadong_data1 = load_tsv_files(huadong_filepath_1)
    huadong_data2 = load_tsv_files(huadong_filepath_2)

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()

    for key in data:
        if group == "all":
            X_train_1, X_test_1, y_train, y_test = preprocess_data(data[key], metadata_path)
            X_h1, y_h1 = preprocess_huadong(huadong_data1[key], metadata_path)
            X_h2, y_h2 = preprocess_huadong(huadong_data2[key], metadata_path)
            X_val_1 = pd.concat([X_h1, X_h2])
            y_val = y_h1 + y_h2
        elif group == 'young' or group == 'old':
            X_train_1, X_test_1, X_val_1, y_train, y_test, y_val = full_preprocessing_y_o_labels(data, huadong_data1,
                                                                                                 huadong_data2, key,
                                                                                                 metadata_path,
                                                                                                 labels_path,
                                                                                                 group)

        X_train = pd.concat([X_train, X_train_1], axis=1)
        X_val = pd.concat([X_val, X_val_1], axis=1)

    if select_features == False:
        file_name = "all_features"

    if select_features == True:
        file_name = "selected_features"
        X_train = select_features_from_paper(X_train, group, key)
        X_val = select_features_from_paper(X_val, group, key)


    common_cols_v = X_train.columns.intersection(X_val.columns).tolist()

    # filling missing values in huadong cohort with zeros
    X_val = X_val.fillna(0)

    # Use the list for indexing
    X_val = X_val[common_cols_v]
    X_train = X_train[common_cols_v]

    # Optional: Add a check to ensure we have common columns
    if not common_cols_v:
        raise ValueError("No common columns found between X_train and X_val")

    # Optional: Log the number of common columns
    logging.info(f"Number of common columns: {len(common_cols_v)}")

    print("number of features: ", X_train.shape[1])
    print("number of samples in training set: ", len(X_train))
    print("number of samples in test set: ", len(X_test))
    print("number of samples in validation set: ", len(X_val))

    return X_train, X_test, X_val, y_train, y_test, y_val


def save_data(
        fudan_filepath: str,
        huadong_filepath_1: str,
        huadong_filepath_2: str,
        labels_filepath: str,
        metadata_filepath: str,
        group: str,
        select_features: bool = False,
        output_dir: Union[str, Path] = None) -> None:

    """
    Save preprocessed training and validation data to CSV files.

    Args:
        fudan_filepath: Path to the FUDAN data file
        huadong_filepath_1: Path to the first part of the Huadong data file
        huadong_filepath_2: Path to the second part of the Huadong data file
        group: Group name for data organization
        select_features: Whether to use selected features or all features
        output_dir: Optional custom output directory. If None, uses Config.DATA_DIR

    Raises:
        OSError: If directory creation fails
        ValueError: If data loading fails
    """
    # Use provided output directory or default to Config.DATA_DIR
    base_dir = Path(output_dir) if output_dir else Path(Config.DATA_DIR)

    # Determine the feature directory name
    feature_dir = "selected_features" if select_features else "all_features"

    # Create the complete save directory path
    save_dir = base_dir / "preprocessed" / group / feature_dir

    try:
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f'Processing data for {feature_dir}...')

        # Load and preprocess the data
        X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data(fudan_filepath, huadong_filepath_1,
        huadong_filepath_2, labels_filepath, metadata_filepath, group, select_features
        )

        # Convert labels to DataFrames
        y_train = pd.DataFrame(y_train)
        y_val = pd.DataFrame(y_val)

        # Define the data pairs to save
        data_pairs = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        }

        # Save all data files
        for name, data in data_pairs.items():
            filepath = save_dir / f"{name}.csv"
            logging.info(f'Saving {name} to {filepath}')
            data.to_csv(filepath)

        logging.info(f'Successfully saved all {feature_dir} data')

    except OSError as e:
        raise OSError(f"Failed to create directory or save files: {e}")
    except Exception as e:
        raise ValueError(f"Error processing data: {e}")



def process_features(
        feature_mode: Literal['all', 'selected', 'both'],
        group_mode: Literal['young', 'old', 'all', 'both'],
        fudan_filepath: str,
        huadong_filepath_1: str,
        huadong_filepath_2: str,
        labels_filepath: str,
        metadata_filepath: str,
        output_dir: Union[str, Path]
) -> None:
    """
    Process data based on feature mode and group selection

    Args:
        feature_mode: Which feature set to process ('all', 'selected', or 'both')
        group_mode: Which group to process ('young', 'old', or 'all')
        fudan_filepath: Path to the FUDAN dataset
        huadong_filepath_1: Path to first Huadong dataset
        huadong_filepath_2: Path to second Huadong dataset
        output_dir: Base directory for output files
    """

    groups_to_process = []
    if group_mode == 'both':
        groups_to_process = ['young', 'old', 'all']
    elif group_mode == 'all':
        groups_to_process = ['all']
    else:
        groups_to_process = [group_mode]  # either 'young' or 'old'

    # Process each group
    for group in groups_to_process:
        logging.info(f'Processing group: {group}')

        # Process features for current group
        if feature_mode in ['all', 'both']:
            logging.info('Processing with all features...')
            save_data(
                fudan_filepath=fudan_filepath,
                huadong_filepath_1=huadong_filepath_1,
                huadong_filepath_2=huadong_filepath_2,
                labels_filepath=labels_filepath,
                metadata_filepath=metadata_filepath,
                group=group,
                select_features=False,
                output_dir=output_dir
            )

        if feature_mode in ['selected', 'both']:
            logging.info('Processing with selected features...')
            save_data(
                fudan_filepath=fudan_filepath,
                huadong_filepath_1=huadong_filepath_1,
                huadong_filepath_2=huadong_filepath_2,
                labels_filepath=labels_filepath,
                metadata_filepath=metadata_filepath,
                group=group,
                select_features=True,
                output_dir=output_dir
            )



class DataPaths(NamedTuple):
    """Container for all data paths used in the script"""
    fudan: Path
    huadong_1: Path
    huadong_2: Path
    labels: Path
    metadata: Path
    output: Path


def setup_logging(log_level: str) -> None:
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_input_paths(paths: DataPaths) -> None:
    """
    Validate that all input files exist

    Args:
        paths: DataPaths object containing all paths

    Raises:
        FileNotFoundError: If any input file is missing
    """
    for name, path in paths._asdict().items():
        if name != 'output':  # Don't check output path as it might not exist yet
            if not path.exists():
                raise FileNotFoundError(f"File not found for {name}: {path}")


def create_output_directory(output_path: Path) -> None:
    """
    Create output directory if it doesn't exist

    Args:
        output_path: Path to output directory
    """
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory verified/created: {output_path}")


def parse_arguments() -> Tuple[DataPaths, str, str, str]:
    """
    Set up argument parser and return validated paths, log level, features mode, and group mode

    Returns:
        Tuple containing DataPaths object, log level string, features mode, and group mode
    """
    parser = argparse.ArgumentParser(description='Process multiple dataset files')

    # Required arguments for all file paths
    parser.add_argument('--fudan_filepath', type=str, required=True,
                        help='Path to the FUDAN dataset file')
    parser.add_argument('--huadong_filepath_1', type=str, required=True,
                        help='Path to the first Huadong dataset file')
    parser.add_argument('--huadong_filepath_2', type=str, required=True,
                        help='Path to the second Huadong dataset file')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to the labels file')
    parser.add_argument('--metadata_filepath', type=str, required=True,
                        help='Path to the metadata file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output files')

    # Add group and features arguments
    parser.add_argument('--group', type=str, default='both',
                        choices=['young', 'old', 'all', 'both'],
                        help='Group for data organization (default: both)')
    parser.add_argument('--features', type=str, default='both',
                        choices=['all', 'selected', 'both'],
                        help='Feature processing mode (default: both)')

    # Optional arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Convert all paths to Path objects
    paths = DataPaths(
        fudan=Path(args.fudan_filepath),
        huadong_1=Path(args.huadong_filepath_1),
        huadong_2=Path(args.huadong_filepath_2),
        metadata=Path(args.metadata_filepath),
        labels=Path(args.labels_path),
        output=Path(args.output_dir)
    )

    return paths, args.log_level, args.features, args.group


def main():
    try:
        # Get and validate paths and arguments
        paths, log_level, features_mode, group_mode = parse_arguments()

        # Setup logging
        setup_logging(log_level)
        logging.info("Starting data processing...")

        # Validate all input paths
        validate_input_paths(paths)
        logging.info("All input files verified")

        # Create output directory
        create_output_directory(paths.output)

        # Process and save data based on feature mode and group mode
        process_features(
            feature_mode=features_mode,
            group_mode=group_mode,
            fudan_filepath=str(paths.fudan),
            huadong_filepath_1=str(paths.huadong_1),
            huadong_filepath_2=str(paths.huadong_2),
            labels_filepath=str(paths.labels),
            metadata_filepath=str(paths.metadata),
            output_dir=paths.output
        )

        logging.info("Processing completed successfully")

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == '__main__':
    main()