from __future__ import annotations

RAW_TARGET_EN = "Churn Value"
TARGET_CN = "\u6d41\u5931\u503c"

CUSTOMER_ID_CN = "\u5ba2\u6237ID"
COUNT_CN = "\u8ba1\u6570"
COUNTRY_CN = "\u56fd\u5bb6"
STATE_CN = "\u5dde"
CITY_CN = "\u57ce\u5e02"
ZIP_CODE_CN = "\u90ae\u653f\u7f16\u7801"
LAT_LONG_CN = "\u7ecf\u7eac\u5ea6\u7ec4\u5408"
LATITUDE_CN = "\u7eac\u5ea6"
LONGITUDE_CN = "\u7ecf\u5ea6"
GENDER_CN = "\u6027\u522b"
SENIOR_CITIZEN_CN = "\u8001\u5e74\u4eba"
PARTNER_CN = "\u4f34\u4fa3"
DEPENDENTS_CN = "\u5bb6\u5c5e"
TENURE_MONTHS_CN = "\u5728\u7f51\u65f6\u957f\uff08\u6708\uff09"
PHONE_SERVICE_CN = "\u7535\u8bdd\u670d\u52a1"
MULTIPLE_LINES_CN = "\u591a\u6761\u7ebf\u8def"
INTERNET_SERVICE_CN = "\u4e92\u8054\u7f51\u670d\u52a1"
ONLINE_SECURITY_CN = "\u7f51\u7edc\u5b89\u5168"
ONLINE_BACKUP_CN = "\u7f51\u7edc\u5907\u4efd"
DEVICE_PROTECTION_CN = "\u8bbe\u5907\u4fdd\u62a4"
TECH_SUPPORT_CN = "\u6280\u672f\u652f\u6301"
STREAMING_TV_CN = "\u6d41\u5a92\u4f53\u7535\u89c6"
STREAMING_MOVIES_CN = "\u6d41\u5a92\u4f53\u7535\u5f71"
CONTRACT_TYPE_CN = "\u5408\u540c\u7c7b\u578b"
PAPERLESS_BILLING_CN = "\u65e0\u7eb8\u5316\u8d26\u5355"
PAYMENT_METHOD_CN = "\u652f\u4ed8\u65b9\u5f0f"
MONTHLY_CHARGES_CN = "\u6708\u8d39\u7528"
TOTAL_CHARGES_CN = "\u603b\u8d39\u7528"
CHURN_LABEL_CN = "\u6d41\u5931\u6807\u7b7e"
CHURN_SCORE_CN = "\u6d41\u5931\u8bc4\u5206"
CLTV_CN = "\u5ba2\u6237\u7ec8\u8eab\u4ef7\u503c"
CHURN_REASON_CN = "\u6d41\u5931\u539f\u56e0"

RAW_TO_CN_RENAME_MAP = {
    "CustomerID": CUSTOMER_ID_CN,
    "Count": COUNT_CN,
    "Country": COUNTRY_CN,
    "State": STATE_CN,
    "City": CITY_CN,
    "Zip Code": ZIP_CODE_CN,
    "Lat Long": LAT_LONG_CN,
    "Latitude": LATITUDE_CN,
    "Longitude": LONGITUDE_CN,
    "Gender": GENDER_CN,
    "Senior Citizen": SENIOR_CITIZEN_CN,
    "Partner": PARTNER_CN,
    "Dependents": DEPENDENTS_CN,
    "Tenure Months": TENURE_MONTHS_CN,
    "Phone Service": PHONE_SERVICE_CN,
    "Multiple Lines": MULTIPLE_LINES_CN,
    "Internet Service": INTERNET_SERVICE_CN,
    "Online Security": ONLINE_SECURITY_CN,
    "Online Backup": ONLINE_BACKUP_CN,
    "Device Protection": DEVICE_PROTECTION_CN,
    "Tech Support": TECH_SUPPORT_CN,
    "Streaming TV": STREAMING_TV_CN,
    "Streaming Movies": STREAMING_MOVIES_CN,
    "Contract": CONTRACT_TYPE_CN,
    "Paperless Billing": PAPERLESS_BILLING_CN,
    "Payment Method": PAYMENT_METHOD_CN,
    "Monthly Charges": MONTHLY_CHARGES_CN,
    "Total Charges": TOTAL_CHARGES_CN,
    "Churn Label": CHURN_LABEL_CN,
    "Churn Value": TARGET_CN,
    "Churn Score": CHURN_SCORE_CN,
    "CLTV": CLTV_CN,
    "Churn Reason": CHURN_REASON_CN,
}

DROP_COLUMNS = [
    COUNTRY_CN,
    STATE_CN,
    CITY_CN,
    ZIP_CODE_CN,
    LAT_LONG_CN,
    LATITUDE_CN,
    LONGITUDE_CN,
    CUSTOMER_ID_CN,
    COUNT_CN,
    CHURN_REASON_CN,
    CHURN_LABEL_CN,
    CHURN_SCORE_CN,
]

NUMERIC_COLUMNS = [
    TENURE_MONTHS_CN,
    MONTHLY_CHARGES_CN,
    TOTAL_CHARGES_CN,
    CLTV_CN,
]

BINARY_MAPPINGS = {
    GENDER_CN: {"Male": 1, "Female": 0},
    SENIOR_CITIZEN_CN: {"Yes": 1, "No": 0},
    PARTNER_CN: {"Yes": 1, "No": 0},
    DEPENDENTS_CN: {"Yes": 1, "No": 0},
    PHONE_SERVICE_CN: {"Yes": 1, "No": 0},
    PAPERLESS_BILLING_CN: {"Yes": 1, "No": 0},
}

FINAL_MODEL_FEATURE_COLUMNS = [
    SENIOR_CITIZEN_CN,
    PARTNER_CN,
    DEPENDENTS_CN,
    TENURE_MONTHS_CN,
    INTERNET_SERVICE_CN,
    ONLINE_SECURITY_CN,
    TECH_SUPPORT_CN,
    STREAMING_TV_CN,
    STREAMING_MOVIES_CN,
    CONTRACT_TYPE_CN,
    PAPERLESS_BILLING_CN,
    PAYMENT_METHOD_CN,
    MONTHLY_CHARGES_CN,
    TOTAL_CHARGES_CN,
    CLTV_CN,
]
