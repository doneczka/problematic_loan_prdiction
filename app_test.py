from locust import HttpUser, task, constant_throughput
import random

test_applications = [
    {
        "AMT_CREDIT": 5000,
        "CODE_GENDER": "F",
        "DAYS_EMPLOYED": -1000,
        "NAME_EDUCATION_TYPE": "Higher education",
        "DAYS_BIRTH": -11000,
        "NAME_FAMILY_STATUS": "Married",
        "NAME_INCOME_TYPE": "Working",
        "FLAG_OWN_CAR": "Y",
        "OWN_CAR_AGE": 5,
        "EXT_SOURCE_1": 0.7,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.6,
        "PAYMENT_RATIO": 1.0,
        "TOTAL_NB_POS_CASH": 5,
        "TOTAL_PAYMENT_AGREEMENT": 30000,
    },
    {
        "AMT_CREDIT": 10000,
        "CODE_GENDER": "M",
        "DAYS_EMPLOYED": -2000,
        "NAME_EDUCATION_TYPE": "Academic degree",
        "DAYS_BIRTH": -11000,
        "NAME_FAMILY_STATUS": "Separated",
        "NAME_INCOME_TYPE": "Working",
        "FLAG_OWN_CAR": "N",
        "OWN_CAR_AGE": 0,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.7,
        "PAYMENT_RATIO": 0.95,
        "TOTAL_NB_POS_CASH": 3,
        "TOTAL_PAYMENT_AGREEMENT": 20000,
    },
]


class ProblematicLoan(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=random.choice(test_applications),
            timeout=5,
        )
