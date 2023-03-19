import requests
import time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Example loan application
    application = {
        "AMT_CREDIT": 5000,
        "CODE_GENDER": "F",
        "DAYS_EMPLOYED": -1000,
        "NAME_EDUCATION_TYPE": "higher",
        "DAYS_BIRTH": -11000,
        "NAME_FAMILY_STATUS": "married",
        "NAME_INCOME_TYPE": "working",
        "FLAG_OWN_CAR": "Y",
        "OWN_CAR_AGE": 5,
        "EXT_SOURCE_1": 0.7,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.6,
        "PAYMENT_RATIO": 1.0,
        "TOTAL_NB_POS_CASH": 5,
        "TOTAL_PAYMENT_AGREEMENT": 30000
    }

    # Location of my server
    url = "https://loan-service-3-sdv3evv4va-nw.a.run.app/predict"

    # Measure the response time
    all_times = []
    # For 100 times
    for i in tqdm(range(100)):
        t0 = time.time_ns() // 1_000_000
        # Send a request
        resp = requests.post(url, json=application)
        t1 = time.time_ns() // 1_000_000
        # Measure how much time it took to get a response in ms
        time_taken = t1 - t0
        all_times.append(time_taken)

    # Print out the results
    print("Response time in ms:")
    print("Median:", np.quantile(all_times, 0.5))
    print("95th percentile:", np.quantile(all_times, 0.95))
    print("Max:", np.max(all_times))
