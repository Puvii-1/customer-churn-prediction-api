"""
Generates a realistic synthetic customer churn dataset. Using a synthetic
dataset (rather than downloading one) keeps the project fully
reproducible and dependency-free — anyone can regenerate the exact same
data with a fixed random seed.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_churn_data(n_customers: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    tenure_months = rng.integers(1, 72, n_customers)
    monthly_charges = rng.uniform(20, 120, n_customers).round(2)
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.9, 1.1, n_customers)).round(2)
    num_support_tickets = rng.poisson(1.5, n_customers)
    contract_type = rng.choice(["month-to-month", "one_year", "two_year"], n_customers, p=[0.55, 0.25, 0.20])
    is_senior_citizen = rng.choice([0, 1], n_customers, p=[0.84, 0.16])
    has_tech_support = rng.choice([0, 1], n_customers, p=[0.5, 0.5])
    payment_delay_days = rng.exponential(2, n_customers).round(1)

    # Build churn probability via a logistic function of standardized risk
    # factors — this gives a clean, learnable signal (unlike a small
    # additive risk model where random noise swamps the real pattern).
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    contract_score = np.where(contract_type == "month-to-month", 1.4,
                      np.where(contract_type == "one_year", -0.3, -1.3))
    tenure_score = -zscore(tenure_months) * 1.1
    support_score = zscore(num_support_tickets) * 0.9
    payment_score = zscore(payment_delay_days) * 0.6
    tech_support_score = np.where(has_tech_support == 1, -0.6, 0.5)

    logit = -1.9 + contract_score + tenure_score + support_score + payment_score + tech_support_score
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_tickets": num_support_tickets,
        "contract_type": contract_type,
        "is_senior_citizen": is_senior_citizen,
        "has_tech_support": has_tech_support,
        "payment_delay_days": payment_delay_days,
        "churned": churned,
    })
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_churn_data()
    df.to_csv("data/churn_data.csv", index=False)
    print(f"Generated {len(df)} rows. Churn rate: {df['churned'].mean():.1%}")