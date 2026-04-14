# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="Demand Intelligence Dashboard")

st.title("🍽️ Demand Intelligence & Kitchen Forecast System")

# -----------------------------
# 📂 Upload Section
# -----------------------------
st.sidebar.header("📂 Upload Data")

file = st.sidebar.file_uploader("Upload Sales File (Excel/CSV)", type=["xlsx", "csv"])

st.sidebar.markdown("""
### Expected Columns:
- Date
- Item Name
- Quantity sold
- Event (optional)
""")

if file:

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

    df['Date'] = pd.to_datetime(df['Date'])

    # Fill missing event
    if 'Event' not in df.columns:
        df['Event'] = 'Normal'
    else:
        df['Event'] = df['Event'].fillna('Normal')

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df['Day'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Day'].isin(['Saturday', 'Sunday'])
    df['Month'] = df['Date'].dt.month

    # Add Weekend as event
    df.loc[df['Is_Weekend'], 'Event'] = df['Event'] + "_Weekend"

    # -----------------------------
    # 📊 OVERVIEW DASHBOARD
    # -----------------------------
    st.header("📊 Overview")

    total_sales = df['Quantity sold'].sum()
    avg_daily = df.groupby('Date')['Quantity sold'].sum().mean()
    total_items = df['Item Name'].nunique()
    volatility = df['Quantity sold'].std()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", int(total_sales))
    col2.metric("Avg Daily Demand", int(avg_daily))
    col3.metric("Unique Items", total_items)
    col4.metric("Volatility", round(volatility, 2))

    # Trend
    st.subheader("📈 Daily Demand Trend")
    trend = df.groupby('Date')['Quantity sold'].sum()
    st.line_chart(trend)

    # Category (Item-based proxy)
    st.subheader("🍛 Item Distribution")
    item_dist = df.groupby('Item Name')['Quantity sold'].sum()
    st.bar_chart(item_dist)

    # Weekend vs Weekday
    st.subheader("📅 Weekday vs Weekend")
    weekend_comp = df.groupby('Is_Weekend')['Quantity sold'].mean()
    st.bar_chart(weekend_comp)

    # Day pattern
    st.subheader("📆 Day of Week Pattern")
    day_pattern = df.groupby('Day')['Quantity sold'].mean()
    st.bar_chart(day_pattern)

    # -----------------------------
    # 📘 EDA STEPS
    # -----------------------------
    st.header("📘 EDA Methodology")

    st.subheader("Baseline Demand Table")

    baseline = df.groupby('Item Name').agg(
        avg_daily=('Quantity sold', 'mean'),
        weekday_avg=('Quantity sold', lambda x: x[df.loc[x.index, 'Is_Weekend']==False].mean()),
        weekend_avg=('Quantity sold', lambda x: x[df.loc[x.index, 'Is_Weekend']==True].mean()),
        volatility=('Quantity sold', 'std')
    )

    baseline['uplift_%'] = ((baseline['weekend_avg'] - baseline['weekday_avg']) / baseline['weekday_avg']) * 100

    st.dataframe(baseline)

    # -----------------------------
    # 🎯 EVENT IMPACT
    # -----------------------------
    st.header("🎯 Event Impact Analysis")

    event_impact = df.groupby('Event')['Quantity sold'].mean()
    normal = event_impact.get('Normal', event_impact.mean())

    impact_df = ((event_impact - normal) / normal) * 100
    impact_df = impact_df.reset_index()
    impact_df.columns = ['Event', '% Change']

    # Classification
    def classify(x):
        if x > 20:
            return "High Impact 📈"
        elif x > 5:
            return "Moderate 📊"
        elif x < -20:
            return "Negative 📉"
        else:
            return "No Impact ⚖️"

    impact_df['Impact Type'] = impact_df['% Change'].apply(classify)

    st.dataframe(impact_df)

    st.bar_chart(impact_df.set_index('Event')['% Change'])

    # -----------------------------
    # 🍛 ITEM ANALYSIS
    # -----------------------------
    st.header("🍛 Item Sensitivity")

    pivot = df.pivot_table(
        values='Quantity sold',
        index='Item Name',
        columns='Event',
        aggfunc='mean'
    )

    st.dataframe(pivot)

    st.subheader("Grouped View")
    st.bar_chart(pivot.fillna(0))

    # -----------------------------
    # 🤖 FORECAST
    # -----------------------------
    st.header("🤖 Forecast")

    df = df.sort_values('Date')
    df['lag1'] = df.groupby('Item Name')['Quantity sold'].shift(1)
    df = df.dropna()

    le_item = LabelEncoder()
    le_day = LabelEncoder()
    le_event = LabelEncoder()

    df['Item_enc'] = le_item.fit_transform(df['Item Name'])
    df['Day_enc'] = le_day.fit_transform(df['Day'])
    df['Event_enc'] = le_event.fit_transform(df['Event'])

    X = df[['Item_enc', 'Day_enc', 'Event_enc', 'lag1']]
    y = df['Quantity sold']

    model = RandomForestRegressor()
    model.fit(X, y)

    st.sidebar.header("🔮 Forecast Input")

    item = st.sidebar.selectbox("Item", df['Item Name'].unique())
    day = st.sidebar.selectbox("Day", df['Day'].unique())
    event = st.sidebar.selectbox("Event", df['Event'].unique())
    lag = st.sidebar.number_input("Previous Day Sales", min_value=0)

    if st.sidebar.button("Predict"):

        pred = model.predict([[le_item.transform([item])[0],
                               le_day.transform([day])[0],
                               le_event.transform([event])[0],
                               lag]])

        st.success(f"Predicted Demand: {int(pred[0])}")

    # Bulk Forecast
    st.subheader("📦 Kitchen Preparation Plan")

    results = []
    for item in df['Item Name'].unique():
        pred = model.predict([[le_item.transform([item])[0],
                               le_day.transform([day])[0],
                               le_event.transform([event])[0],
                               lag]])
        results.append([item, int(pred[0])])

    result_df = pd.DataFrame(results, columns=["Item", "Predicted Demand"])
    st.dataframe(result_df)

    # -----------------------------
    # 🚨 SURPRISE DAYS
    # -----------------------------
    st.header("🚨 Surprise Days")

    mean = df['Quantity sold'].mean()
    std = df['Quantity sold'].std()

    df['Z'] = (df['Quantity sold'] - mean) / std

    df['Anomaly'] = df['Z'].apply(lambda x: "Spike" if x > 1.8 else ("Drop" if x < -1.8 else "Normal"))

    anomalies = df[df['Anomaly'] != "Normal"]

    st.dataframe(anomalies[['Date', 'Item Name', 'Quantity sold', 'Anomaly', 'Event']])

else:
    st.info("👆 Upload your dataset to begin analysis")