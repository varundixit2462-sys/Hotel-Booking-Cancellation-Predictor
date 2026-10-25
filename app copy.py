"""
Hotel Booking Cancellation Predictor - Streamlit app

Run on your computer:   streamlit run app.py
Files this app needs in the same folder:
    hotel_cancel_model.joblib   (trained pipeline, saved by the notebook)
    model_meta.json             (feature list + dropdown choices, saved by the notebook)
"""
import json

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")


# ------------------------------------------------------------------ load the model once
# @st.cache_resource keeps the model in memory, so it is not reloaded on every click
@st.cache_resource
def load_model():
    model = joblib.load("hotel_cancel_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_model()
FEATURES = meta["numeric_features"] + meta["categorical_features"]
CHOICES = meta["categories"]
DEFAULTS = meta["numeric_defaults"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MEAL_LABELS = {"BB": "BB - Bed & breakfast", "HB": "HB - Half board",
               "FB": "FB - Full board", "SC": "SC - No meal"}


def risk_band(prob, threshold):
    """Turn a probability into a label and a suggested action."""
    if prob >= max(threshold, 0.7):
        return "High risk", "Ask for a deposit or a confirmation call before the arrival date."
    if prob >= threshold:
        return "Medium risk", "Send a reminder email two weeks before arrival."
    return "Low risk", "No action needed."


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Flag a booking as 'will cancel' when the probability is at least",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower = catch more cancellations but more false alarms. Higher = fewer, surer flags.",
    )
    st.divider()
    st.subheader("About the model")
    m = meta["test_metrics"]
    st.write("Gradient boosting classifier trained on "
             f"{meta['training_rows']:,} real hotel bookings.")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy']:.0%}")
    c2.metric("ROC AUC", f"{m['roc_auc']:.2f}")
    c1.metric("Precision", f"{m['precision']:.0%}")
    c2.metric("Recall", f"{m['recall']:.0%}")
    st.caption("Test-set scores at a 0.5 threshold.")

# ------------------------------------------------------------------ header
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Enter the details of a new booking to estimate how likely it is to be cancelled.")

tab_single, tab_batch, tab_help = st.tabs(["Single booking", "Many bookings (CSV)", "How it works"])

# ------------------------------------------------------------------ tab 1: one booking
with tab_single:
    with st.form("booking_form"):
        st.subheader("Stay")
        a, b, c = st.columns(3)
        hotel = a.selectbox("Hotel", CHOICES["hotel"])
        arrival_month = b.selectbox("Arrival month", MONTHS, index=7)
        lead_time = c.number_input("Lead time (days between booking and arrival)",
                                   min_value=0, max_value=800, value=int(DEFAULTS["lead_time"]))
        a, b, c = st.columns(3)
        weekend_nights = a.number_input("Weekend nights", 0, 20, 1)
        week_nights = b.number_input("Week nights", 0, 50, 2)
        adr = c.number_input("Price per night (€)", min_value=0.0, max_value=1000.0,
                             value=round(DEFAULTS["adr"], 2), step=5.0)

        st.subheader("Guests")
        a, b, c, d = st.columns(4)
        adults = a.number_input("Adults", 0, 10, 2)
        children = b.number_input("Children", 0, 10, 0)
        babies = c.number_input("Babies", 0, 10, 0)
        country = d.selectbox("Country (ISO code)", CHOICES["country"],
                              help="PRT = Portugal, GBR = UK, FRA = France, ESP = Spain, DEU = Germany")
        a, b, c, d = st.columns(4)
        repeated = a.checkbox("Has stayed before")
        prev_cancel = b.number_input("Previous cancellations", 0, 30, 0)
        prev_ok = c.number_input("Previous completed stays", 0, 80, 0)
        customer_type = d.selectbox("Customer type", CHOICES["customer_type"])

        st.subheader("Booking details")
        a, b, c = st.columns(3)
        market_segment = a.selectbox("Market segment", CHOICES["market_segment"])
        channel = b.selectbox("Distribution channel", CHOICES["distribution_channel"])
        deposit = c.selectbox("Deposit type", CHOICES["deposit_type"])
        a, b, c = st.columns(3)
        meal = a.selectbox("Meal plan", CHOICES["meal"], format_func=lambda x: MEAL_LABELS.get(x, x))
        room = b.selectbox("Room type reserved", sorted(CHOICES["reserved_room_type"]))
        waiting = c.number_input("Days on the waiting list", 0, 400, 0)
        a, b = st.columns(2)
        requests = a.number_input("Special requests", 0, 5, 0)
        parking = b.number_input("Parking spaces requested", 0, 3, 0)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Build a one-row table with EXACTLY the column names the model was trained on
        booking = pd.DataFrame([{
            "lead_time": lead_time, "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights, "adults": adults, "children": children,
            "babies": babies, "is_repeated_guest": int(repeated),
            "previous_cancellations": prev_cancel, "previous_bookings_not_canceled": prev_ok,
            "days_in_waiting_list": waiting, "adr": adr, "required_car_parking_spaces": parking,
            "total_of_special_requests": requests, "hotel": hotel, "arrival_date_month": arrival_month,
            "meal": meal, "country": country, "market_segment": market_segment,
            "distribution_channel": channel, "reserved_room_type": room,
            "deposit_type": deposit, "customer_type": customer_type,
        }])[FEATURES]

        if adults + children + babies == 0:
            st.warning("A booking needs at least one guest.")
        else:
            prob = model.predict_proba(booking)[0, 1]
            label, action = risk_band(prob, threshold)

            st.divider()
            left, right = st.columns([1, 2])
            left.metric("Chance of cancellation", f"{prob:.0%}")
            right.progress(float(prob))
            if label == "Low risk":
                right.success(f"**{label}.** {action}")
            elif label == "Medium risk":
                right.warning(f"**{label}.** {action}")
            else:
                right.error(f"**{label}.** {action}")

# ------------------------------------------------------------------ tab 2: CSV upload
with tab_batch:
    st.write("Upload a CSV file with one booking per row. It must contain these columns:")
    st.code(", ".join(FEATURES), language=None)
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing_cols = [col for col in FEATURES if col not in data.columns]
        if missing_cols:
            st.error(f"These columns are missing: {', '.join(missing_cols)}")
        else:
            data["cancel_probability"] = model.predict_proba(data[FEATURES])[:, 1].round(3)
            data["flagged"] = data["cancel_probability"] >= threshold
            n_flag = int(data["flagged"].sum())
            st.success(f"Scored {len(data)} bookings. {n_flag} flagged at a {threshold:.2f} threshold.")
            st.dataframe(data.sort_values("cancel_probability", ascending=False))
            st.download_button("Download results as CSV", data.to_csv(index=False),
                               file_name="scored_bookings.csv", mime="text/csv")

# ------------------------------------------------------------------ tab 3: explanation
with tab_help:
    st.markdown(
        """
**Data.** About 87,000 unique bookings (after cleaning) from a city hotel and a resort hotel in
Portugal, July 2015 to August 2017. Source: Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*.

**Model.** A scikit-learn pipeline: numbers are scaled, categories are one-hot encoded, and a
histogram gradient boosting classifier produces the probability.

**What raises the risk.** Booking far in advance, booking from Portugal (the hotels' home country),
group and offline travel-agent bookings, and no special requests.

**What lowers it.** Special requests, parking, short lead times, and repeat guests.

**Threshold.** Use the slider in the sidebar. A low threshold suits a cheap action (a reminder email);
a high threshold suits a costly one (asking for a deposit).
        """
    )
