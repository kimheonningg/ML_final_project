import pandas as pd
from typing import Optional, Callable
import numpy as np

t_f_inverse_y = Callable[[np.array], np.array]


def _check_valid_feature_engineering(
    original_train_dataset: pd.DataFrame,
    original_test_dataset: pd.DataFrame,
    augmented_train_dataset: pd.DataFrame,
    augmented_test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
    inverse_y: t_f_inverse_y = lambda x: x,
) -> None:
    assert "y" in augmented_train_dataset.columns, "y must be in train_dataset"
    assert "y" in augmented_test_dataset.columns, "y must be in test_dataset"

    assert isinstance(augmented_train_dataset, pd.DataFrame) and isinstance(
        augmented_test_dataset, pd.DataFrame
    ), "augmented_train_dataset and augmented_test_dataset must be pandas DataFrames"
    if max_column is not None:
        assert (
            augmented_train_dataset.shape[1] <= max_column + 1
            and augmented_test_dataset.shape[1] <= max_column + 1
        ), (
            "max_column is greater than the number of columns in train_dataset and test_dataset"
        )
    assert np.allclose(
        original_train_dataset["y"].to_numpy(),
        inverse_y(augmented_train_dataset["y"].to_numpy()),
        atol=1e-6,
    ), "inverse_y must be a function that recovers the original y"
    assert np.allclose(
        original_test_dataset["y"].to_numpy(),
        inverse_y(augmented_test_dataset["y"].to_numpy()),
        atol=1e-6,
    ), "inverse_y must be a function that recovers the original y"


def template_feature_engineering(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )

    return augmented_train_dataset, augmented_test_dataset, inverse_y


################################################################################################
#
# Example Feature Engineering
#
################################################################################################
def fill_nan_with_mean(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    numeric_cols = augmented_train_dataset.select_dtypes(
        include=["float64", "int64"]
    ).columns
    mean_values = augmented_train_dataset[numeric_cols].mean()

    augmented_train_dataset[numeric_cols] = augmented_train_dataset[
        numeric_cols
    ].fillna(mean_values)
    augmented_test_dataset[numeric_cols] = augmented_test_dataset[numeric_cols].fillna(
        mean_values
    )

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )

    return augmented_train_dataset, augmented_test_dataset, inverse_y


def remove_random_columns(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    import random

    columns_to_remove = list(augmented_train_dataset.columns.drop("y"))
    column_to_remove = random.sample(columns_to_remove, 1)[0]

    augmented_train_dataset = augmented_train_dataset.drop(columns=column_to_remove)
    augmented_test_dataset = augmented_test_dataset.drop(columns=column_to_remove)

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )
    return augmented_train_dataset, augmented_test_dataset, inverse_y


def add_random_columns(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    for i in range(3):
        col_name = f"random_noise_{i}"
        augmented_train_dataset[col_name] = np.random.normal(
            0, 1, size=len(augmented_train_dataset)
        )
        augmented_test_dataset[col_name] = np.random.normal(
            0, 1, size=len(augmented_test_dataset)
        )

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )

    return augmented_train_dataset, augmented_test_dataset, inverse_y


def rolling_window_feature(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    numeric_cols = augmented_train_dataset.select_dtypes(
        include=["float64", "int64"]
    ).columns
    numeric_cols = numeric_cols.drop("y") if "y" in numeric_cols else numeric_cols

    for col in numeric_cols:
        col_name = f"{col}_rolling_mean_3d"
        augmented_train_dataset[col_name] = (
            augmented_train_dataset[col].rolling(window=3, min_periods=1).mean()
        )
        augmented_test_dataset[col_name] = (
            augmented_test_dataset[col].rolling(window=3, min_periods=1).mean()
        )

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )

    return augmented_train_dataset, augmented_test_dataset, inverse_y


def add_one_all_columns(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    augmented_train_dataset = augmented_train_dataset + 1
    augmented_test_dataset = augmented_test_dataset + 1

    def inverse_y(x: np.array) -> np.array:
        return x - 1

    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )
    return augmented_train_dataset, augmented_test_dataset, inverse_y


################################################################################################
#
#           Implement your own feature engineering
#
################################################################################################


def feature_engineering_experiment_1(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: pd.DataFrame) -> pd.DataFrame:
        return x

    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################

    numeric_cols = augmented_train_dataset.select_dtypes(include=np.number).columns
    if 'y' in numeric_cols:
        numeric_cols = numeric_cols.drop('y')

    if not numeric_cols.empty:
        scaler_mean = augmented_train_dataset[numeric_cols].mean()
        scaler_std = augmented_train_dataset[numeric_cols].std()

        scaler_std[scaler_std == 0] = 1

        augmented_train_dataset[numeric_cols] = (augmented_train_dataset[numeric_cols] - scaler_mean) / scaler_std
        augmented_test_dataset[numeric_cols] = (augmented_test_dataset[numeric_cols] - scaler_mean) / scaler_std

    # TODO: implement your own feature engineering after removing this line
    ############################################################################################
    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )
    return augmented_train_dataset, augmented_test_dataset, inverse_y


def feature_engineering_experiment_2(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################

    numeric_cols = augmented_train_dataset.select_dtypes(include=np.number).columns
    if 'y' in numeric_cols:
        numeric_cols = numeric_cols.drop('y')

    if not numeric_cols.empty:
        feature_to_square = numeric_cols[0]

        augmented_train_dataset[f'{feature_to_square}_sq'] = augmented_train_dataset[feature_to_square]**2
        augmented_test_dataset[f'{feature_to_square}_sq'] = augmented_test_dataset[feature_to_square]**2

    # TODO: implement your own feature engineering after removing this line
    ############################################################################################
    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )
    return augmented_train_dataset, augmented_test_dataset, inverse_y


def feature_engineering_experiment_3(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(x: np.array) -> np.array:
        return x

    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################

    original_train_len = len(augmented_train_dataset)
    combined_df = pd.concat([augmented_train_dataset, augmented_test_dataset], ignore_index=True)

    combined_df['y_lag_1'] = combined_df['y'].shift(1)

    combined_df['y_lag_1'] = combined_df['y_lag_1'].fillna(0)
    
    augmented_train_dataset = combined_df.iloc[:original_train_len].copy()
    augmented_test_dataset = combined_df.iloc[original_train_len:].copy()

    # TODO: implement your own feature engineering after removing this line
    ############################################################################################
    _check_valid_feature_engineering(
        train_dataset,
        test_dataset,
        augmented_train_dataset,
        augmented_test_dataset,
        max_column,
        inverse_y,
    )
    return augmented_train_dataset, augmented_test_dataset, inverse_y


def final_feature_engineering(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    max_column: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, t_f_inverse_y]:
    assert isinstance(train_dataset, pd.DataFrame) and isinstance(
        test_dataset, pd.DataFrame
    ), "train_dataset and test_dataset must be pandas DataFrames"
    augmented_train_dataset = train_dataset.copy()
    augmented_test_dataset = test_dataset.copy()

    def inverse_y(y_pred: np.array) -> np.array:
        return y_pred

    ############################################################################################
    #
    #                   Feel free to modify the code below.
    #
    ############################################################################################
    for col in [f'feat_{i}' for i in range(1, 9)]:
        augmented_train_dataset[col] = augmented_train_dataset[col].ffill()
        augmented_test_dataset[col] = augmented_test_dataset[col].ffill()

    for df in [augmented_train_dataset, augmented_test_dataset]:
        df.drop(columns=['feat_7', 'feat_8'], inplace=True)

    for lag in range(1, 4): 
        augmented_train_dataset[f'y_lag_{lag}'] = augmented_train_dataset['y'].shift(lag)
        augmented_test_dataset[f'y_lag_{lag}'] = augmented_test_dataset['y'].shift(lag)

    for lag in range(1, 4):
        augmented_train_dataset[f'y_lag_{lag}'] = augmented_train_dataset[f'y_lag_{lag}'].bfill()
        augmented_test_dataset[f'y_lag_{lag}'] = augmented_test_dataset[f'y_lag_{lag}'].bfill()

    y_values = augmented_train_dataset['y']
    fft_values = np.fft.fft(y_values)

    for i in range(1, 3): 
        augmented_train_dataset[f'fft_real_{i}'] = np.real(fft_values[i]) / len(y_values)
        augmented_train_dataset[f'fft_imag_{i}'] = np.imag(fft_values[i]) / len(y_values)
        augmented_test_dataset[f'fft_real_{i}'] = np.real(fft_values[i]) / len(y_values)
        augmented_test_dataset[f'fft_imag_{i}'] = np.imag(fft_values[i]) / len(y_values)

    features_to_keep = [
        'day_of_week', 'hour', 'minute',
        'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6',
        'y_lag_1', 'y_lag_2', 'y_lag_3',
        'fft_real_1', 'fft_imag_1', 'fft_real_2'
    ]


    augmented_train_dataset = augmented_train_dataset[features_to_keep + ['y']]
    augmented_test_dataset = augmented_test_dataset[features_to_keep + ['y']]

    if max_column is not None:
        features = [col for col in augmented_train_dataset.columns if col != 'y']
        if len(features) > max_column:
            features_to_keep = features[:max_column]
            augmented_train_dataset = augmented_train_dataset[features_to_keep + ['y']]
            augmented_test_dataset = augmented_test_dataset[features_to_keep + ['y']]
    
    # TODO: implement your own feature engineering after removing this line
    ############################################################################################

    # _check_valid_feature_engineering(
    #     original_train_dataset, original_test_dataset,
    #     final_train_dataset, final_test_dataset,
    #     max_column=30, inverse_y=final_inverse_y
    # )

    return augmented_train_dataset, augmented_test_dataset, inverse_y
