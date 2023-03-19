from fastapi import FastAPI
from pydantic import BaseModel, validator
import pickle
import pandas as pd

app = FastAPI()

model_name = "Problem Loan Prediction"


class LoanApplication(BaseModel):
    AMT_CREDIT: float
    CODE_GENDER: str
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str
    DAYS_BIRTH: int
    NAME_FAMILY_STATUS: str
    NAME_INCOME_TYPE: str
    FLAG_OWN_CAR: str
    OWN_CAR_AGE: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    PAYMENT_RATIO: float
    TOTAL_NB_POS_CASH: float
    TOTAL_PAYMENT_AGREEMENT: float

    @validator("AMT_CREDIT")
    def amt_credit_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("AMT_CREDIT must be greater than 0")
        return v

    @validator("CODE_GENDER")
    def code_gender_must_be_valid(cls, v):
        if v not in {"M", "F"}:
            raise ValueError("CODE_GENDER must be M or F")
        return v

    @validator("DAYS_EMPLOYED")
    def days_employed_must_be_negative(cls, v):
        if v >= 0:
            raise ValueError("DAYS_EMPLOYED must be negative")
        return v

    @validator("DAYS_BIRTH")
    def days_birth_must_be_negative(cls, v):
        if v >= 0:
            raise ValueError("DAYS_BIRTH must be negative")
        return v

    @validator("NAME_INCOME_TYPE")
    def name_income_type_must_be_valid(cls, v):
        allowed_values = [
            "Working",
            "State servant",
            "Commercial associate",
            "Pensioner",
            "Unemployed",
            "Student",
            "Businessman",
            "Maternity leave",
        ]
        if v not in allowed_values:
            raise ValueError(
                f'Invalid NAME_INCOME_TYPE: {v}. Allowed values: {", ".join(allowed_values)}'
            )
        return v

    @validator("NAME_EDUCATION_TYPE")
    def name_education_type_must_be_valid(cls, v):
        allowed_values = [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ]
        if v not in allowed_values:
            raise ValueError(
                f'Invalid NAME_EDUCATION_TYPE: {v}. Allowed values: {", ".join(allowed_values)}'
            )
        return v

    @validator("NAME_FAMILY_STATUS")
    def name_family_status_must_be_valid(cls, v):
        allowed_values = [
            "Single / not married",
            "Married",
            "Civil marriage",
            "Widow",
            "Separated",
            "Unknown",
        ]
        if v not in allowed_values:
            raise ValueError(
                f'Invalid NAME_FAMILY_STATUS: {v}. Allowed values: {", ".join(allowed_values)}'
            )
        return v

    @validator("EXT_SOURCE_1")
    def ext_source_1_must_be_valid(cls, v):
        if v < 0 or v > 1:
            raise ValueError("EXT_SOURCE_1 must be between 0 and 1")
        return v

    @validator("EXT_SOURCE_2")
    def ext_source_2_must_be_valid(cls, v):
        if v < 0 or v > 1:
            raise ValueError("EXT_SOURCE_2 must be between 0 and 1")
        return v

    @validator("EXT_SOURCE_3")
    def ext_source_3_must_be_valid(cls, v):
        if v < 0 or v > 1:
            raise ValueError("EXT_SOURCE_3 must be between 0 and 1")
        return v

    @validator("FLAG_OWN_CAR")
    def flag_own_car_must_be_valid(cls, v):
        if v not in {"Y", "N"}:
            raise ValueError("FLAG_OWN_CAR must be Y or N")
        return v

    @validator("OWN_CAR_AGE")
    def own_car_age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("OWN_CAR_AGE must be greater or equal to 0")
        return v

    @validator("PAYMENT_RATIO")
    def payment_ratio_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("PAYMENT_RATIO must be greater than 0")
        return v

    @validator("TOTAL_NB_POS_CASH")
    def total_nb_poc_cash_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("TOTAL_NB_POS_CASH must be greater or equal to 0")
        return v

    @validator("TOTAL_PAYMENT_AGREEMENT")
    def total_payment_agreement_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("TOTAL_PAYMENT_AGREEMENT must be greater or equal to 0")
        return v


class SuccessOutput(BaseModel):
    prediction: str
    probability: str


model = pickle.load(open("problem_loan.pkl", "rb"))


@app.get("/")
def model_info():
    """Return model information
    XGBoost for classyfying if someone may have troubles repaying its loans.

    It returns 1 when someone is most likely to exhibit payment problems
    and 0 if someone is likely to repay their loans on time."""
    return {"name": model_name}


@app.get("/health")
def service_health():
    """Return service health"""
    return {"ok"}


@app.post("/predict", response_model=SuccessOutput)
def get_prediction(inputs: LoanApplication):
    df = pd.DataFrame([inputs.dict()])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    max_value = probabilities.max() * 100
    if prediction == 0:
        prediction = "No payment difficulites"
    else:
        prediction = "Alert!"
    return {"prediction": prediction, "probability": f"{max_value:.2f} %"}
