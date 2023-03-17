from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from feature_engine.imputation import (
    MeanMedianImputer,
    CategoricalImputer,
    AddMissingIndicator,
)
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from feature_engine.outliers import Winsorizer


class DataPreprocessor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.numeric_features = None
        self.categorical_features = None

    def change_datatypes(self):
        flag_cols = [col for col in self.X.columns if "FLAG" in col]
        self.X[flag_cols] = self.X[flag_cols].astype(str)
        not_cols = [col for col in self.X.columns if "NOT" in col]
        self.X[not_cols] = self.X[not_cols].astype(str)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=0, stratify=self.y
        )

    def define_feature_types(self):
        self.numeric_features = self.X_train.select_dtypes(
            exclude="object"
        ).columns.tolist()
        self.categorical_features = self.X_train.select_dtypes(
            include="object"
        ).columns.tolist()

    def preprocess_data(self):
        self.change_datatypes()
        self.split_data()
        self.define_feature_types()

    def fit(self, X_train):
        num_pipeline = Pipeline(
            [
                ("missing_indicator", AddMissingIndicator()),
                (
                    "drop_constant",
                    DropConstantFeatures(tol=0.9, missing_values="ignore"),
                ),
                (
                    "smart_corr",
                    SmartCorrelatedSelection(method="pearson", threshold=0.8),
                ),
                ("winsorizer", Winsorizer(tail="both", missing_values="ignore")),
                (
                    "imputer",
                    MeanMedianImputer(imputation_method="median", variables=None),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                ("missing_indicator", AddMissingIndicator()),
                (
                    "drop_constant",
                    DropConstantFeatures(tol=0.9, missing_values="ignore"),
                ),
                ("imputer", CategoricalImputer(fill_value="unknown", variables=None)),
                ("encoder", CountFrequencyEncoder(variables=None)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numeric_features),
                ("cat", cat_pipeline, self.categorical_features),
            ]
        )

        self.preprocessor.fit(self.X_train)

    def transform(self, X_train, X_test):
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        return X_train_processed, X_test_processed
    
    def get_feature_names_out(self):
        output_features = []
        for name, pipe, features in self.preprocessor.transformers_:
            if name != 'remainder':
                for i in pipe.get_feature_names_out(features):
                    output_features.append(i)
            else:
                output_features.extend(features)
        return output_features
