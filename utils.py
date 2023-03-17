# Basic data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import float64
import shap
import pickle
from typing import List, Tuple


# Models
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce
from category_encoders.ordinal import OrdinalEncoder

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Functions
from utils import *

import warnings

warnings.simplefilter("ignore")

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


def create_dataframe(
    name_of_feature: str, name_of_dataframe: str, column: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns dataframe with calculated %of total population from a given column

    ----
    Arguments:
    name_of_feature: str
        name of the chosen feature stored for a purpose of further manipulation
    name_of_dataframe: str
        name of the dataframe with a data
    column: str
        name of the chosen feature
    df: pd.DataFrame
        pandas dataframe with features

    ---
    Return:
    pd.DataFrame
        dataframe with two columns: one with the original feature and their values
        and the other with %of total population from a given column
    """

    column_name = name_of_feature
    name_of_dataframe = pd.DataFrame()
    name_of_dataframe["nr_of_people"] = (
        df[column].value_counts().sort_values(ascending=False)[:10]
    )

    name_of_dataframe["%_total"] = round(
        (name_of_dataframe["nr_of_people"]) * 100 / len(df[column]), 2
    )
    name_of_dataframe = name_of_dataframe.reset_index().rename(
        columns={"index": f"{column_name}"}
    )
    name_of_dataframe = name_of_dataframe.append(
        {
            f"{column_name}": "Other",
            "nr_of_people": (
                (len(df[column])) - (name_of_dataframe["nr_of_people"].sum())
            ),
            "%_total": 100 - (name_of_dataframe["%_total"].sum()),
        },
        ignore_index=True,
    )
    return name_of_dataframe


def get_redundant_pairs(df: pd.DataFrame) -> set:
    """Get diagonal and lower triangular pairs of correlation matrix

    ----
    Arguments:
    df: pd.DataFrame
        pandas dataframe with features

    ---
    Return:
    pairs_to_drop: set
        set of correlated features
    """

    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df: pd.DataFrame, n: int) -> pd.Series:
    """Calculates correlation between features and returns top N (chosen) pairs

    ----
    Arguments:
    df: pd.DataFrame
        pandas dataframe with features
    n: int
        number of desired top correlated pairs of features

    ---
    Return:
    au_corr: pd.Series
        pandas Series with pairs of correlated features and values of their correlation
    """

    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def create_preprocessor(num_features: List, cat_features: List) -> ColumnTransformer:
    """Returns ColumnTransformer with processing steps for both categorical and numerical features.

    ----
    Arguments:
    num_features: List
        List of the names of the numerical features
    cat_features: List
        List of the names of the categorical features

    ---
    Return:
    preprocessor: ColumnTransformer
        ColumnTransformer for all features
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OrdinalEncoder(drop_invariant=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    return preprocessor


def visualize_predictions(
    model: Pipeline, X_test: np.array, y_test: np.array, lc=LabelEncoder
) -> pd.DataFrame:
    """Returns classification report as a styled pandas dataframe.
     ----
    Arguments:
    model: Pipeline
        Machine Learning model that is already fitted
    X_test: np.array
        numpy array with test features values
    y_test: np.array
        numpy array with test target values
    lc: LabelEncoder
        encoder for target variable, already fitted

    ---
    Return:
    df: pd.DataFrame
        pandas dataframe with classification report (precision, recall, f1-score and support)
    """

    y_pred = model.predict(X_test)
    df = pd.DataFrame(
        classification_report(
            y_test, y_pred, output_dict=True, target_names=lc.classes_
        )
    ).T.style.background_gradient(cmap="Blues")
    return df


def plot_countplots(data: pd.DataFrame, col: str, hue: str) -> plt.Figure:
    """
    Returns graph with countplot for a chosen categorical variable.

    ----
    Arguments:
    data: pd.DataFrame
        pandas dataframe with features
    col: str
        name of the chosen feature
    hue: str
        name of the target variable

    ----
    Return:
    plt.Figure
        countplot for a given feature
    """

    plt.figure(figsize=(10, 8))
    sns.catplot(
        data=data,
        x=col,
        hue=hue,
        kind="count",
        palette="pastel",
        edgecolor=".6",
        order=["A", "B", "C", "D", "E", "F", "G"],
    )
    plt.title(f"Distribuition of loan grades and {hue}")


def diff_in_distr(df: pd.DataFrame, list_of_columns: List, hue: str) -> plt.Figure:
    """
    Visualize differences in data distribution taking into account target variable.

    ----
    Arguments:
    data: pd.DataFrame
        pandas dataframe with features
    list_of_columns: List
        list of features to visualize
    hue: str
        name of the target variable

    ---
    Return:
    plt.Figure
        countplots of a featurse with target variable as hue
    """
    maxn = 7
    fig, axes = plt.subplots(3, 4, figsize=(26, 16))
    for i, ax in enumerate(axes.flatten()):
        sns.countplot(
            data=df, x=df[list_of_columns[i]], hue=hue, ax=ax, palette="seismic"
        )
        sns.despine(left=True)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        ax.set_xlabel("")
        ax.set_xlim(-0.5, maxn - 0.5)
        ax.set_title(f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
        ax.legend(loc="upper right")

    plt.tight_layout()


def normalized_cm(y_test: np.array, y_pred: np.array, model_name: str) -> plt.Figure:
    """
    Returns normalized by support values confusion matrix.

    Arguments:
    y_test: np.array
        array of y_test labels
    y_pred: np.array
        array of y_pred labels
    model_name: str
        name of the models that made predictions

    Returns:
    plt.Figure: confusion matrix
    """

    cm = confusion_matrix(y_test, y_pred)
    normalized_cm = np.round((cm / np.sum(cm, axis=1).reshape(-1, 1)), 3)
    sns.heatmap(normalized_cm, annot=True, cmap="Greens")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.title(f"Normalized confusion matrix for {model_name}")


def report_as_df(y_test: np.array, y_pred: np.array) -> pd.DataFrame:
    """Returns classification report in a form of dataframe.

    Arguments:
    y_test: np.array
        array of y_test labels
    y_pred: np.array
        array of y_pred labels

    Returns:
    pd.DataFrame: dataframe with classification report.
    """

    df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).T.style.background_gradient(cmap="Blues")
    return df


def create_boxplots(df: pd.DataFrame, n: int, m: int, figsize: Tuple) -> plt.Figure:
    """
    Visualize distribution of numerical features from a given dataframe.

    ----
    Arguments:
    df: pd.DataFrame
        pandas dataframe with features
    n: int
        number of rows
    m: int
        number of columns
    figsize: Tuple
        size of the figure

    ---
    Return:
    plt.Figure
        set of boxplots
    """

    inputs = list(df.columns[2:])
    fig, axs = plt.subplots(n, m, figsize=figsize)
    for i, (ax, curve) in enumerate(zip(axs.flat, inputs)):
        sns.boxplot(
            y=df[curve],
            ax=ax,
            color="cornflowerblue",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": "6",
            },
            flierprops={"marker": "o", "markeredgecolor": "darkgreen"},
        )

        ax.set_title('')
        ax.set_ylabel("")
        ax.set_xlabel(inputs[i])

    plt.subplots_adjust(hspace=0.25, wspace=0.75)
    plt.tight_layout()
    plt.show()

def boxplots_with_target(df: pd.DataFrame, n: int, m: int, target: str, figsize: Tuple) -> plt.Figure:
    """
    Visualize distribution of numerical features from a given dataframe.

    ----
    Arguments:
    df: pd.DataFrame
        pandas dataframe with features
    n: int
        number of rows
    m: int
        number of columns
    target: str
        target variable
    figsize: Tuple
        size of the figure

    ---
    Return:
    plt.Figure
        set of boxplots
    """
    inputs = list(df.columns[2:])
    fig, axs = plt.subplots(n, m, figsize=figsize)
    for i, (ax, curve) in enumerate(zip(axs.flat, inputs)):
        sns.boxplot(
            y=df[curve],
            x=df[target],
            ax=ax,
            flierprops={"marker": "o", "markeredgecolor": "darkgrey"},
        )

        ax.set_title(inputs[i], y=1.05)
        ax.set_ylabel("")

    plt.subplots_adjust(hspace=0.25, wspace=0.75)
    plt.tight_layout()
    plt.show()


def plot_histograms(df: pd.DataFrame, n: int, m: int, figsize: Tuple, hue:str) -> plt.Figure:
    """
    Visualize distribution of numerical features with histograms from a given dataframe.

    ----
    Arguments:
    df: pd.DataFrame
        pandas dataframe with features
    n: int
        number of rows
    m: int
        number of columns
    figsize: Tuple
        size of the figure
    hue: str
        name of the target variable

    ---
    Return:
    plt.Figure
        set of histograms
    """
    inputs = list(df.columns[2:])
    fig, axs = plt.subplots(n, m, figsize=figsize)
    for i, (ax, curve) in enumerate(zip(axs.flat, inputs)):
        sns.histplot(x=df[curve], ax=ax, hue=hue, stat='density', common_norm=False)
        ax.set_title(inputs[i], y=1.1)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.subplots_adjust(hspace=0.25, wspace=0.75)
    plt.tight_layout()
    plt.show()


def create_heatmap(df: pd.DataFrame, title: "str", figsize: Tuple) -> plt.Figure:
    """
    Visualize correlation between features in a dataframe.

    ----
    Arguments:
    data: pd.DataFrame
        pandas dataframe with features
    title: str
        name of the dataframe
    figsize: Tuple
        size of the figure

    ---
    Return:
    plt.Figure
        correlation matrix
    """

    corr_matrix = df.iloc[:, 2:].corr(method="pearson")
    sns.set(font_scale=1)
    f, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="YlGnBu",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.5},
    )
    f.tight_layout()
    ax.set_title(
        f"Correlation heatmap of {title}",
        fontdict={"fontsize": 16},
        pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")


def plot_pred_vs_real(pred: np.array, test: np.array) -> plt.Figure:
    """Returns graph with distribution plot for predictions and real data.

     ----
    Arguments:
    pred: np.array
        numpy array with values of predictions
    test: np.array
        numpy array with values of test data

    ---
    Return:
    plt.Figure
        distribution plot for predictions and test data.
    """

    plt.figure(figsize=(12, 6))
    sns.distplot(pred, color="blue", label="Distrib Predictions", hist=False)
    sns.distplot(test, color="red", label="Distrib Original", hist=False)
    plt.title("Distribution of pred and original interest rate")
    plt.legend()


def check_object_values(data: pd.DataFrame, n: int, m: int):
    """Returns information about specific object type column:
    it's name, unique values, number of unique values and number of missing values.

    """
    for column in data.iloc[:, n:m]:
        print(column)
        print(data[column].unique())
        print(f"Number of unique classes: {len(data[column].unique())}")
        print(f"Number of missing values: {data[column].isnull().sum()}")
        print("\n")


def show_column_description(df: pd.DataFrame, table_name: str) -> pd.DataFrame():
    return (
        df[df["Table"] == f"{table_name}.csv"][["Row", "Description", "Special"]]
        .style.set_properties(**{"text-align": "left"})
        .set_table_styles([dict(selector="th", props=[("text-align", "left")])])
    )
