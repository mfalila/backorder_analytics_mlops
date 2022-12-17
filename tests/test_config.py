#from prediction_service.prediction import form_response, api_response
import prediction_service.prediction
import prediction_service

import pandas as pd
import pytest


input_data = {
    "incorrect_range":
        {"national_inv": 100,
         "lead_time": 10,
         "in_transit_qty": 20,
         "forecast_3_month": 30,
         "sales_3_month": 11,
         "perf_6_month_avg": 0.99,
         "deck_risk_Yes": 1,
         "neg_inv_balance": 0},

    "incorrect_values":
        {"national_inv": 100,
         "lead_time": 10,
         "in_transit_qty": "as",
         "forecast_3_month": 30,
         "sales_3_month": 11,
         "perf_6_month_avg": 0.99,
         "deck_risk_Yes": "ab",
         "neg_inv_balance": "no"},

    "correct_range":
        {"national_inv": 100,
         "lead_time": 10,
         "in_transit_qty": 10,
         "forecast_3_month": 30,
         "sales_3_month": 11,
         "perf_6_month_avg": 0.99,
         "deck_risk_Yes": 0,
         "neg_inv_balance": 1},

    "incorrect_col":
        {"national_inv": 100,
         "lead time": 10,
         "in_transit_qty": 10,
         "forecast_3_month": 30,
         "sales_3_month": 11,
         "perf_6_month_avg": 0.99,
         "deck_risk_Yes": 0,
         "neg_inv_balance": 1}
}

TARGET_range = {
    "min": 0,
    "max": 1
}


def test_form_response_correct_range(sample_dict=input_data["correct_range"]):
    sample_list = [sample_dict]  # convert to list
    sample_df = pd.DataFrame(data=sample_list)
    res = prediction_service.prediction.form_response(sample_df)
    assert TARGET_range["min"] <= res[0, 0] <= TARGET_range["max"]


def test_api_response_correct_range(sample_dict=input_data["correct_range"]):
    res = prediction_service.prediction.api_response(sample_dict)
    # res = float((([x for x in res.values()])[0]).strip('"'))
    assert TARGET_range["min"] <= res <= TARGET_range["max"]


def test_form_response_incorrect_range(sample_dict=input_data["incorrect_range"]):
    sample_list = [sample_dict]  # convert to list
    sample_df = pd.DataFrame(data=sample_list)
    res = prediction_service.prediction.form_response(sample_df)
    pytest.raises(prediction_service.prediction.NotInRange)


def test_api_response_incorrect_range(sample_dict=input_data["incorrect_range"]):
    res = prediction_service.prediction.api_response(sample_dict)
    assert res["response"] == prediction_service.prediction.NotInRange().message


def test_api_response_incorrect_col(sample_dict=input_data["incorrect_col"]):
    res = prediction_service.prediction.api_response(sample_dict)
    assert res["response"] == prediction_service.prediction.NotInCols().message
