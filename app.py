import streamlit as st
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from plot_utils import style_streamlit_plot
# This logic must come AFTER the import streamlit line
# --- FIX THIS SECTION ---
# Line 14
if not os.path.exists('random_forest.pkl'):
    with st.spinner("Training model..."):
        subprocess.run(["python", "train_models.py"]) # <--- MUST HAVE 4 SPACES HERE
subprocess.run(["python", "train_models.py"])

# Change this:
df = pd.read_csv('hotel_bookings.csv')# ------------------- Page config -------------------
st.set_page_config(page_title="StaySerene Predictor", layout="wide")

# ------------------- Background styling (restored) -------------------
background_image_url = "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=1920&q=80"
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                    url("{background_image_url}");
        background-size: cover;
    }}
    .main-title {{
        font-size: 2.2rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 0.2rem;
    }}
    .sub-title {{
        font-size: 1.1rem;
        color: #34495E;
        margin-bottom: 1.5rem;
    }}
    .metric-cell {{
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #2C3E50;
    }}
    .result-text {{
        font-size: 1.2rem;
        font-weight: 700;
        color: #1A2B3C;
    }}
    .result-box {{
        background-color: rgba(255,255,255,0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }}
    div[data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}
    .stButton button {{
        border-radius: 20px;
        font-weight: 500;
    }}

</style>
""", unsafe_allow_html=True)


# ------------------- Load artifacts (hotel models) -------------------
@st.cache_resource
def load_hotel_models():
    # Remove 'models/' from all four lines below
    rf = joblib.load('random_forest.pkl') 
    xgb = joblib.load('xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return rf, xgb, scaler, columns

try:
    rf, xgb, scaler, columns = load_hotel_models()
    models_loaded = True
except FileNotFoundError:
    st.error("Model files not found. Please run `src/train_models.py` first.")
    models_loaded = False
    st.stop()

# ------------------- Hotel metrics (from your notebook) -------------------
hotel_metrics = {
    'Random Forest': {'Accuracy': 0.8897, 'Precision': 0.8778, 'Recall': 0.8161, 'F1': 0.8458},
    'XGBoost': {'Accuracy': 0.8733, 'Precision': 0.8004, 'Recall': 0.8771, 'F1': 0.8370}
}

# ------------------- Navigation (top) -------------------
nav_col1, nav_col2, nav_col3, _ = st.columns([1, 1, 1, 3])
with nav_col1:
    predictor_clicked = st.button("🏨 Predictor", use_container_width=True)
with nav_col2:
    analysis_clicked = st.button("📊 Data Analysis", use_container_width=True)
with nav_col3:
    new_analysis_clicked = st.button("📈 New Analysis", use_container_width=True)

if 'page' not in st.session_state:
    st.session_state.page = 'predictor'

if predictor_clicked:
    st.session_state.page = 'predictor'
if analysis_clicked:
    st.session_state.page = 'analysis'
if new_analysis_clicked:
    st.session_state.page = 'new_analysis'

# ------------------- Load cleaned dataset for analysis pages -------------------
from src.data_preprocessing import load_and_clean_data

df = pd.read_csv('hotel_bookings.csv')
# ===================== PAGE: PREDICTOR =====================
if st.session_state.page == 'predictor':
    st.markdown('<p class="main-title">StaySerene – Cancellation Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Select a model, review its performance, and predict new bookings.</p>',
                unsafe_allow_html=True)

    # Model selection
    st.markdown("#### Choose Model")
    model_choice = st.radio("", ["Random Forest", "XGBoost"], horizontal=True, label_visibility="collapsed")
    selected_model = rf if model_choice == "Random Forest" else xgb

    # Performance table
    st.markdown(f"#### {model_choice} Performance")
    m = hotel_metrics[model_choice]
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"<div class='metric-cell'>{m['Accuracy']:.2%}</div><div class='metric-label'>Accuracy</div>",
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-cell'>{m['Precision']:.2%}</div><div class='metric-label'>Precision</div>",
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='metric-cell'>{m['Recall']:.2%}</div><div class='metric-label'>Recall</div>",
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='metric-cell'>{m['F1']:.2%}</div><div class='metric-label'>F1 Score</div>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # --- Prediction result (displayed ABOVE the form if exists) ---
    if 'pred_result' in st.session_state and st.session_state.pred_result is not None:
        proba, pred = st.session_state.pred_result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Result")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"<p class='result-text'>Cancellation Probability: <b>{proba:.1%}</b></p>",
                        unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<p class='result-text' style='color:#B03A2E;'>⚠️ <b>Likely to cancel</b></p>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<p class='result-text' style='color:#1E8449;'>✅ <b>Likely to stay</b></p>",
                            unsafe_allow_html=True)
        with res_col2:
            st.info("💡 **Top factors** – Feature importance coming soon.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Booking form
    st.markdown("## Booking Details")
    with st.form("booking_form"):
        row1 = st.columns(3)
        with row1[0]:
            lead_time = st.number_input("Lead Time (days)", 0, 700, 100)
        with row1[1]:
            adr = st.number_input("Average Daily Rate (ADR)", 0.0, 500.0, 100.0, step=5.0)
        with row1[2]:
            total_guests = st.number_input("Total Guests", 1, 20, 2)

        row2 = st.columns(3)
        with row2[0]:
            hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        with row2[1]:
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        with row2[2]:
            total_stays = st.number_input("Stay Nights", 0, 50, 3)

        row3 = st.columns(3)
        with row3[0]:
            booking_changes = st.slider("Booking Changes", 0, 5, 0)
        with row3[1]:
            previous_cancellations = st.slider("Previous Cancellations", 0, 10, 0)
        with row3[2]:
            market_segment = st.selectbox("Market Segment",
                                          ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups",
                                           "Complementary", "Aviation"])

        row4 = st.columns(3)
        with row4[0]:
            customer_type = st.selectbox("Customer Type",
                                         ["Transient", "Contract", "Group", "Transient-Party"])
        with row4[1]:
            required_parking = st.slider("Car Parking Spaces", 0, 5, 0)
        with row4[2]:
            special_requests = st.slider("Special Requests", 0, 5, 0)

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            submitted = st.form_submit_button("🔮 Bookings", use_container_width=True)
        with btn_col2:
            reset = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submitted:
            # Build input dict
            input_dict = {
                'lead_time': lead_time,
                'adr': adr,
                'total_guests': total_guests,
                'total_stays': total_stays,
                'booking_changes': booking_changes,
                'previous_cancellations': previous_cancellations,
                'required_car_parking_spaces': required_parking,
                'total_of_special_requests': special_requests,
                'hotel_City Hotel': 1 if hotel == "City Hotel" else 0,
                'deposit_type_Non Refund': 1 if deposit_type == "Non Refund" else 0,
                'deposit_type_Refundable': 1 if deposit_type == "Refundable" else 0,
                'market_segment_Corporate': 1 if market_segment == "Corporate" else 0,
                'market_segment_Direct': 1 if market_segment == "Direct" else 0,
                'market_segment_Groups': 1 if market_segment == "Groups" else 0,
                'market_segment_Offline TA/TO': 1 if market_segment == "Offline TA/TO" else 0,
                'market_segment_Online TA': 1 if market_segment == "Online TA" else 0,
                'market_segment_Aviation': 1 if market_segment == "Aviation" else 0,
                'market_segment_Complementary': 1 if market_segment == "Complementary" else 0,
                'customer_type_Contract': 1 if customer_type == "Contract" else 0,
                'customer_type_Group': 1 if customer_type == "Group" else 0,
                'customer_type_Transient': 1 if customer_type == "Transient" else 0,
                'customer_type_Transient-Party': 1 if customer_type == "Transient-Party" else 0,
            }
            input_df = pd.DataFrame([input_dict]).reindex(columns=columns, fill_value=0)
            num_cols = ['lead_time', 'adr', 'total_guests', 'total_stays']
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            proba = selected_model.predict_proba(input_df)[0][1]
            pred = selected_model.predict(input_df)[0]

            # Store result in session state
            st.session_state.pred_result = (proba, pred)
            st.rerun()  # Rerun to show result above form immediately

        if reset:
            st.session_state.pred_result = None
            st.rerun()

# ===================== DATA ANALYSIS PAGE =====================
if st.session_state.get("page") == "analysis":
    st.markdown("<h2 style='color: #2C3E50;'>📊 Dataset Overview & Balance</h2>", unsafe_allow_html=True)

    # 1. DATA IMBALANCE VISUALIZATION
    col_bal1, col_bal2 = st.columns([1, 1.5])
    with col_bal1:
        st.write("**Target Variable Balance**")
        counts = df['is_canceled'].value_counts()
        labels = ['Stayed (0)', 'Canceled (1)']

        fig_bal, ax_bal = plt.subplots(figsize=(4, 4))
        ax_bal.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,
                   colors=['#2ECC71', '#E74C3C'], explode=(0.05, 0))
        fig_bal = style_streamlit_plot(fig_bal, ax_bal)
        st.pyplot(fig_bal, transparent=True)

    with col_bal2:
        st.warning("**DATA IMBALANCE OBSERVATION:**")
        # Fixed the multi-line string by keeping it on one line or using triple quotes
        st.write("""
        **The dataset shows a split of ~63% Stays vs ~37% Cancellations.** 
        **To prevent the model from being biased toward successful stays, we have implemented 
        balanced class weights in our training scripts to ensure high sensitivity to cancellation risks.**
        """)

        total_loss = df[df['is_canceled'] == 1]['financial_loss'].sum()
        st.metric(label="💸 Total Potential Financial Loss", value=f"${total_loss:,.2f}")

    # ===================== DATA ANALYSIS PAGE (Rules Section) =====================
    st.markdown("---")
    st.markdown("### 🧬 Cancellation Patterns (Association Rules)")
    st.write("These 'If-Then' patterns identify the highest-risk combinations found in canceled bookings.")

    if os.path.exists('cancellation_rules.csv'):
        rules = pd.read_csv('cancellation_rules.csv').head(6)

        # Display rules in a 2-column grid
        r_cols = st.columns(2)
        for i, row in rules.iterrows():
            col_idx = i % 2
            # Clean the text (remove frozenset, brackets, and underscores)
            ant = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "").replace("_",
                                                                                                                " ")
            con = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "").replace("_",
                                                                                                                " ")

            with r_cols[col_idx]:
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-left: 5px solid #E74C3C; 
                            border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <b style="color: #7F8C8D; font-size: 0.8rem;">PATTERN #{i + 1}</b><br>
                    <p style="margin: 5px 0;"><b>IF:</b> {ant}</p>
                    <p style="margin: 5px 0; color: #E74C3C;"><b>THEN:</b> {con}</p>
                    <small style="color: #95A5A6;">Lift: {row['lift']:.2f} | Confidence: {row['confidence']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        # This is what you are seeing now because the file is missing
        st.info("💡 Association Rules not found. Please run 'python -m src.train_models' to generate insights.")
    # 2. HOTEL TYPE & CUSTOMER TYPE
    st.markdown("<h3 style='color: #2E86C1;'>🔍 General Segment Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cancellation by Hotel Type**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df.groupby('hotel')['is_canceled'].mean().plot(kind='bar', color=['#3498DB', '#9B59B6'], ax=ax1)
        st.pyplot(style_streamlit_plot(fig1, ax1), transparent=True)
        st.info("**INSIGHT:** **City Hotels face higher cancellation rates compared to Resort Hotels.**")

    with col2:
        st.write("**Cancellation by Customer Type**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df.groupby('customer_type')['is_canceled'].mean().plot(kind='bar', color='#1ABC9C', ax=ax2)
        st.pyplot(style_streamlit_plot(fig2, ax2), transparent=True)
        st.info("**INSIGHT:** **Transient customers show the highest volatility in bookings.**")
    st.markdown("---")

    with st.expander("📊 Cancellation Pattern Insights (Association Rules)"):
            try:
                rules_df = pd.read_csv('hotel_bookings.csv')
                # Format for display
                display_rules = rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
                st.dataframe(display_rules)
                st.caption("Rules with lift > 1 indicate strong associations among canceled bookings.")
            except FileNotFoundError:
                st.warning("Association rules file not found. Please run the association mining notebook first.")
# ===================== PAGE: NEW ANALYSIS =====================
elif st.session_state.page == "new_analysis":
    st.markdown("<h2 style='color: #2C3E50;'>📈 Advanced Trends & Hotspots</h2>", unsafe_allow_html=True)

    # 1. Yearly Trends
    st.write("**Yearly Cancellation Volume**")
    fig_year, ax_year = plt.subplots(figsize=(10, 3))
    yearly_cancels = df[df['is_canceled'] == 1]['arrival_date_year'].value_counts().sort_index()
    ax_year.plot(yearly_cancels.index.astype(str), yearly_cancels.values, marker='o', color='#E74C3C', linewidth=3)
    ax_year.fill_between(yearly_cancels.index.astype(str), yearly_cancels.values, color='#E74C3C', alpha=0.2)
    st.pyplot(style_streamlit_plot(fig_year, ax_year), transparent=True)

    st.warning(f"**YEARLY TREND:** **The peak year for cancellations was {yearly_cancels.idxmax()}.**")

    st.markdown("---")

    # 2. Monthly Hotspots
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Cancellations by Month**")
        fig_month, ax_month = plt.subplots(figsize=(5, 4))
        df_canceled = df[df['is_canceled'] == 1]
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                       'November', 'December']
        monthly_cancels = df_canceled['arrival_date_month'].value_counts().reindex(month_order)
        ax_month.bar(monthly_cancels.index, monthly_cancels.values, color='#8E44AD')
        plt.xticks(rotation=45)
        st.pyplot(style_streamlit_plot(fig_month, ax_month), transparent=True)
    with col_b:
        st.success("**MONTHLY ANALYSIS:** **High-volume months suggest times for stricter deposit policies.**")

    st.markdown("---")

    # 3. DEPOSIT & MARKET SEGMENT
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Impact of Deposit Type**")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        df.groupby('deposit_type')['is_canceled'].mean().plot(kind='bar', color=['#E67E22', '#E74C3C', '#2ECC71'],
                                                              ax=ax3)
        st.pyplot(style_streamlit_plot(fig3, ax3), transparent=True)
        st.success("**POLICY:** **'Non-Refund' policies effectively eliminate cancellation risk.**")

    with col4:
        st.write("**Top Cancellation Market Segments**")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        df.groupby('market_segment')['is_canceled'].mean().sort_values().plot(kind='barh', color='#34495E', ax=ax4)
        st.pyplot(style_streamlit_plot(fig4, ax4), transparent=True)
        st.success("**MARKET:** **Online Travel Agents (TA) contribute the most to cancellations.**")

    # 4. VIP STATUS
    st.markdown("---")
    col_vip1, col_vip2 = st.columns([1, 1])
    with col_vip1:
        st.write("**VIP vs Standard Cancellation**")
        fig_vip, ax_vip = plt.subplots(figsize=(5, 4))
        df.groupby('customer_vip_status')['is_canceled'].mean().plot(kind='bar', color=['#34495E', '#D4AC0D'],
                                                                     ax=ax_vip)
        st.pyplot(style_streamlit_plot(fig_vip, ax_vip), transparent=True)
    with col_vip2:
        st.info("**VIP ANALYSIS:** **VIP guests are significantly more loyal and less likely to cancel.**")

