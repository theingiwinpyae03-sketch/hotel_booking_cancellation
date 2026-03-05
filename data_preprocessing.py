import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # ---- your exact cleaning steps from the notebook ----
    # Deleting unnecessary columns
    df.drop('company', axis=1, inplace=True)
    df.drop(['reservation_status', 'reservation_status_date'], axis=1, inplace=True)

    # Handling missing values
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')

    # Country grouping
    top4 = df['country'].value_counts().head(4).index
    df['country'] = df['country'].apply(lambda x: x if x in top4 else 'Other')

    # Feature engineering
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df = df[df['total_guests'] > 0]

    # Financial loss (needed for later, will be dropped before training)
    df['financial_loss'] = df.apply(
        lambda x: x['adr'] * x['total_stays'] if x['is_canceled'] == 1 else 0, axis=1
    )

    # VIP status (optional – you can keep it)
    def assign_vip(row):
        if row['is_repeated_guest'] == 1 or row['adr'] > 250 or row['total_of_special_requests'] > 2:
            return 'VIP'
        return 'Standard'
    df['customer_vip_status'] = df.apply(assign_vip, axis=1)

    return df