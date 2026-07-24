const form = document.getElementById("churn-form");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");

const gaugeFill = document.getElementById("gauge-fill");
const gaugeValue = document.getElementById("gauge-value");
const riskBadge = document.getElementById("risk-badge");
const riskBadgeText = document.getElementById("risk-badge-text");
const recommendation = document.getElementById("recommendation");

const GAUGE_CIRCUMFERENCE = 251.2; // matches stroke-dasharray in CSS, for a half-circle arc of this radius

const RISK_COPY = {
  low: "Low risk. No action needed right now.",
  medium: "Some risk signals present. Consider a check-in with this customer.",
  high: "High risk. Recommend proactive outreach within the next couple of days.",
};

function setGauge(probability, riskLevel) {
  const pct = Math.round(probability * 100);
  const offset = GAUGE_CIRCUMFERENCE * (1 - probability);

  gaugeFill.style.strokeDashoffset = offset;
  gaugeValue.textContent = `${pct}%`;

  riskBadge.classList.remove("low", "medium", "high");
  riskBadge.classList.add(riskLevel);
  riskBadgeText.textContent = `${riskLevel.toUpperCase()} RISK`;

  recommendation.textContent = RISK_COPY[riskLevel] || "";
}

function resetGauge() {
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  gaugeValue.textContent = "—";
  riskBadge.classList.remove("low", "medium", "high");
  riskBadgeText.textContent = "Awaiting input";
  recommendation.textContent = "Fill in the account details and predict to see a risk assessment and recommended action.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMsg.textContent = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  const payload = {
    tenure_months: Number(document.getElementById("tenure_months").value),
    monthly_charges: Number(document.getElementById("monthly_charges").value),
    total_charges: Number(document.getElementById("total_charges").value),
    num_support_tickets: Number(document.getElementById("num_support_tickets").value),
    contract_type: document.getElementById("contract_type").value,
    is_senior_citizen: document.getElementById("is_senior_citizen").checked ? 1 : 0,
    has_tech_support: document.getElementById("has_tech_support").checked ? 1 : 0,
    payment_delay_days: Number(document.getElementById("payment_delay_days").value),
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const detail = Array.isArray(errData.detail)
        ? errData.detail.map((d) => d.msg).join(", ")
        : errData.detail || "Something went wrong. Check the values and try again.";
      throw new Error(detail);
    }

    const data = await response.json();
    setGauge(data.churn_probability, data.risk_level);
  } catch (err) {
    errorMsg.textContent = err.message;
    resetGauge();
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict risk";
  }
});