"""
Feature engineering pipeline that takes in 'clean' customer data
and transform into 'features' for modelling
"""

import pandas as pd
import numpy as np

def fe_customer(input_data_orig):
    # Generate tenure_months feature
    input_data = input_data_orig\
        .assign(
            tenure_months=lambda df: 
            (df.days_since_first_order - df.days_since_last_order) / 30.0
        )
    input_data['tenure_months'] = np.ceil(input_data.tenure_months)
    input_data['tenure_months'] = input_data['tenure_months'].replace(0.0, 1.0)

    # convert transactional features into monthly
    features_numeric_trasactional = input_data[[
        'orders','items','cancels','returns', 'vouchers',
        'female_items','male_items','unisex_items','wapp_items','wftw_items',
        'mapp_items','wacc_items','macc_items','mftw_items','wspt_items','mspt_items',
        'curvy_items','sacc_items',
        'msite_orders','desktop_orders','android_orders','ios_orders','other_device_orders',
        'work_orders','home_orders','parcelpoint_orders','other_collection_orders',
        'redpen_discount_used','coupon_discount_applied',
        'revenue',
        ]].fillna(0.0)
    features_numeric_monthly = (1.0 * features_numeric_trasactional)\
        .div(input_data.tenure_months, axis='index')

    # generate other numeric features
    features_numeric_other = input_data[[
        'days_since_first_order','days_since_last_order','tenure_months',
                'different_addresses','shipping_addresses','devices',
            'average_discount_onoffer','average_discount_used',
        ]]
    features_numeric_other = features_numeric_other * 1.0

    # generate the categorical features
    features_categorical = input_data[[
        'is_newsletter_subscriber', 'cc_payments','paypal_payments',
        'afterpay_payments','apple_payments'
        ]]
    features_categorical = features_categorical\
        .assign(is_newsletter_subscriber=lambda df: 
            (df.is_newsletter_subscriber == 'Y') * 1.0
            )
    features_categorical = 1.0 * features_categorical

    # Combine everything
    output_data = pd.concat([
            input_data[['customer_id']], 
            features_categorical, 
            features_numeric_monthly, 
            features_numeric_other
        ], 
            axis = 1
        )
    output_data = output_data.fillna(0.0)

    return output_data
