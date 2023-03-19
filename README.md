# problematic_loan_prdiction

## **GOAL** – prediction of which person might struggle with loan repayment
-	Find personal traits of people who might not pay on time their debts
-	Fine-tune the model to get the best results
- Model deployment on GCP

## **Methods and techniques:**
-	EDA to check what kind of data I deal with – internal/external data; current applications for credits, previous credit history; borrower characteristics.
-	Fature engineering with feature-engine library
-	Introduction of classes for preprocessing and models
-	RandomizedSearchCV for hyperparameter tuning 
-	Boruta feature selection technique
-	Feature importance with shap library
-	Docker for contenerization
-	FastAPI and GCP for deployment
-	Locust for testing

## **Conclusions:**

COST_CALCULATION - According to my simulation, if we replace the current model, which predicts bad loans with a 60% true positive rate, with my solution that predicts bad loans with a 68.7% true positive rate, we could minimize the loss from bad predictions by about 19%, from 12.4 million with the current solution to 10.1 million with my solution. 

Additionally, by alerting predictions with pred_proba of class 0 ranging from 0.5 to 0.65 based on my model, we could potentially minimize loss by another 5 million. 

Regarding my investigation of the Home Credit Group's datasets, there were a total of eight datasets, including the application train/test dataset, two credit bureau datasets, and four previous borrower application datasets.

 For my risk assessment strategy, I used the main dataset, application train, which included the 'TARGET' column indicating which applications were considered risky. After experimenting with various models and feature engineering techniques, I found that the XGBClassifier model had the best performance with an AUC score of 0.752 on the test set. 

Subsenquentially, I added features from the credit bureau datasets and created a 'TRUST INDICATOR' column to determine which applicants were more likely to pay on time or not. However, adding these columns did not increase the ROC-AUC score. 

Next, I selected only accepted applications from the previous_application and installments_payments datasets and added a 'PAYMENT_RATIO' column, which proved to be a significant predictor, increasing the ROC-AUC score to 0.758. 

Then, I added features from the credit_card_balance and pos_cash_balance datasets, but neither improved the results. 

Finally, I performed Boruta feature selection and chose 15 most important features for my final model, including three engineered features: 'PAYMENT_RATIO', 'TOTAL_PAYMENT_AGREEMENT', and 'TOTAL_NB_POS_CASH'. The final ROC-AUC score was 0.752. 

Overall, my investigation suggests that my solution could improve the current model's performance, resulting in significant cost savings. 

However, further experimentation and refinement of the model may be necessary to achieve even better results. 

Deployment capabilities: 
> ● Testing with basic python script that sends request to ML model’s API endpoint:
> 
> Response time in ms: Median: 118.0 
> 
> 95th percentile: 140.1 Max: 21902 
> 
> ● Testing with Locust: 100 users/swarn rate-2: 
>
>Median: 480 
>
>95th percentile: 5100 Max: 5218 
>
>current failures: 21.4
