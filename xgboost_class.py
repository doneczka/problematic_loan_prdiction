import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display
import scikitplot as skplt
import shap


class XGBoostClassifier:

    def __init__(self, **params):
        self.params = params
        self.clf = xgboost.XGBClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        proba = self.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        y_predict_proba = self.predict_proba(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))

        skplt.metrics.plot_roc_curve(y, y_predict_proba, ax=axes[0])
        axes[0].set_title("ROC Curve")
            
        c_m = confusion_matrix(y, pred)
        normalized_cm = np.round((c_m / np.sum(c_m, axis=1).reshape(-1, 1)), 3)
        sns.heatmap(normalized_cm, annot=True, cmap="Greens", ax=axes[1])
        axes[1].set_xlabel("Predicted values")
        axes[1].set_ylabel("Actual values")
        axes[1].set_title("Normalized Confusion Matrix")
        
        cr = classification_report(y, pred, output_dict=True)
        df = pd.DataFrame(cr).T.style.background_gradient(cmap="Blues")
        display(df)

        plt.show()
        
        print(f"AUC on a test set: {auc:.3f}")

    def random_search(self, X, y, param_distributions, n_iter=15, cv=3, n_jobs=-1):
        rs = RandomizedSearchCV(self.clf, param_distributions=param_distributions, n_iter=n_iter, cv=cv, n_jobs=n_jobs, scoring='roc_auc', random_state=42)
        rs.fit(X, y)
        print(f"Best parameters: {rs.best_params_}")
        print(f"Best AUC: {rs.best_score_:.4f}")
        self.clf = rs.best_estimator_

    def explain(self, X_train, feature_names):
        explainer = shap.TreeExplainer(self.clf)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
        plt.title('Shap values for 20 most important features')
        plt.show()

    def predict_problematic_loan(self, X):
        """
        Predict the loan decision and probabilities for a given input data X, 
        which should contain the loan amount feature.
        Returns a pandas DataFrame containing the loan amount, the loan decision,
        and the probabilities of the classes.
        """
        # extract the loan amount feature from the input data
        loan_amount = X[:, 0].reshape(-1, 1)

        # make predictions and get probabilities for each class
        pred = self.predict(X)
        proba = self.predict_proba(X)

        # create a DataFrame with the loan amount and predicted class
        df = pd.DataFrame(loan_amount, columns=['loan_amount'])
        df['loan_decision'] = pred

        # add the probabilities of each class to the DataFrame
        classes = self.classes_
        for i, cls in enumerate(classes):
            df[f'probability_{cls}'] = proba[:, i]

        return df[['loan_amount', 'loan_decision', 'probability_0', 'probability_1']]
    