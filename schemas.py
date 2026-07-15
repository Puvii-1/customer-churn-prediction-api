from pydantic import BaseModel, Field
from typing import Literal


class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total amount billed to date")
    num_support_tickets: int = Field(..., ge=0, description="Support tickets raised")
    contract_type: Literal["month-to-month", "one_year", "two_year"]
    is_senior_citizen: Literal[0, 1]
    has_tech_support: Literal[0, 1]
    payment_delay_days: float = Field(..., ge=0, description="Average payment delay in days")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 5,
                "monthly_charges": 89.50,
                "total_charges": 447.50,
                "num_support_tickets": 4,
                "contract_type": "month-to-month",
                "is_senior_citizen": 0,
                "has_tech_support": 0,
                "payment_delay_days": 6.5,
            }
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool