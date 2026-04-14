import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="Demand Intelligence Dashboard")

st.title("🍽️ Demand Intelligence & Kitchen Forecast System")

# -----------------------------
# Upload
# -----------------------------
st.sidebar.header("📂 Upload Data")
file = st.sidebar.file_uploader("Upload Sales File (Excel/CSV)", type=["xlsx", "csv"])

if file:

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

    # ✅ STANDARDIZE COLUMN NAMES (ONLY ONCE)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ✅ AUTO-MAP COMMON COLUMN VARIATIONS
    col_map = {
        'date': 'date',
        'item_name': 'item_name',
        'item': 'item_name',
        'product': 'item_name',
        'quantity_sold': 'quantity_sold',
        'qty': 'quantity_sold',
        'quantity': 'quantity_sold',
        'event': 'event'
    }

    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    # ✅ CHECK REQUIRED
    required = ['date', 'item_name', 'quantity_sold']
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df['date'] = pd.to_datetime(df['date'])

    if 'event' not in df.columns:
        df['event'] = 'normal'
    else:
        df['event'] = df['event'].fillna('normal')

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df['day'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday'])
    df['month'] = df['date'].dt.month

    df.loc[df['is_weekend'], 'event'] = df['event'] + "_weekend"

    # -----------------------------
    # Overview
    # -----------------------------
    st.header("📊 Overview")

    total_sales = df['quantity_sold'].sum()
    avg_daily = df.groupby('date')['quantity_sold'].sum().mean()
    total_items = df['item_name'].nunique()
    volatility = df['quantity_sold'].std()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", int(total_sales))
    c2.metric("Avg Daily", int(avg_daily))
    c3.metric("Items", total_items)
    c4.metric("Volatility", round(volatility, 2))

    # Charts
    st.line_chart(df.groupby('date')['quantity_sold'].sum())
    st.bar_chart(df.groupby('item_name')['quantity_sold'].sum())
    st.bar_chart(df.groupby('is_weekend')['quantity_sold'].mean())
    st.bar_chart(df.groupby('day')['quantity_sold'].mean())

    # -----------------------------
    # EDA
    # -----------------------------
    st.header("📘 EDA")

    baseline = df.groupby('item_name').agg(
        avg_daily=('quantity_sold', 'mean'),
        weekday_avg=('quantity_sold', lambda x: x[df.loc[x.index, 'is_weekend']==False].mean()),
        weekend_avg=('quantity_sold', lambda x: x[df.loc[x.index, 'is_weekend']==True].mean()),
        volatility=('quantity_sold', 'std')
    )

    baseline['uplift_%'] = ((baseline['weekend_avg'] - baseline['weekday_avg']) / baseline['weekday_avg']) * 100
    st.dataframe(baseline)

    # -----------------------------
    # Event Impact
    # -----------------------------
    st.header("🎯 Event Impact")

    event_impact = df.groupby('event')['quantity_sold'].mean()
    normal = event_impact.get('normal', event_impact.mean())

    impact = ((event_impact - normal) / normal) * 100
    impact = impact.reset_index()
    impact.columns = ['event', '% change']

    st.dataframe(impact)
    st.bar_chart(impact.set_index('event')['% change'])

    # -----------------------------
    # Item Sensitivity
    # -----------------------------
    st.header("🍛 Item Sensitivity")

    pivot = df.pivot_table(
        values='quantity_sold',
        index='item_name',
        columns='event',
        aggfunc='mean'
    )

    st.dataframe(pivot)
    st.bar_chart(pivot.fillna(0))

    # -----------------------------
    # Forecast
    # -----------------------------
    st.header("🤖 Forecast")

    df = df.sort_values('date')
    df['lag1'] = df.groupby('item_name')['quantity_sold'].shift(1)
    df = df.dropna()

    le_item = LabelEncoder()
    le_day = LabelEncoder()
    le_event = LabelEncoder()

    df['item_enc'] = le_item.fit_transform(df['item_name'])
    df['day_enc'] = le_day.fit_transform(df['day'])
    df['event_enc'] = le_event.fit_transform(df['event'])

    X = df[['item_enc', 'day_enc', 'event_enc', 'lag1']]
    y = df['quantity_sold']

    model = RandomForestRegressor()
    model.fit(X, y)

    st.sidebar.header("🔮 Forecast")

    item = st.sidebar.selectbox("Item", df['item_name'].unique())
    day = st.sidebar.selectbox("Day", df['day'].unique())
    event = st.sidebar.selectbox("Event", df['event'].unique())
    lag = st.sidebar.number_input("Previous Day Sales", min_value=0)

    if st.sidebar.button("Predict"):
        pred = model.predict([[le_item.transform([item])[0],
                               le_day.transform([day])[0],
                               le_event.transform([event])[0],
                               lag]])
        st.success(f"Predicted Demand: {int(pred[0])}")

    # Bulk
    results = []
    for i in df['item_name'].unique():
        p = model.predict([[le_item.transform([i])[0],
                            le_day.transform([day])[0],
                            le_event.transform([event])[0],
                            lag]])
        results.append([i, int(p[0])])

    st.dataframe(pd.DataFrame(results, columns=["item", "prediction"]))

    # -----------------------------
    # Anomaly Detection
    # -----------------------------
    st.header("🚨 Surprise Days")

    mean = df['quantity_sold'].mean()
    std = df['quantity_sold'].std()

    df['z'] = (df['quantity_sold'] - mean) / std

    df['anomaly'] = df['z'].apply(
        lambda x: "Spike" if x > 1.8 else ("Drop" if x < -1.8 else "Normal")
    )

    anomalies = df[df['anomaly'] != "Normal"]

    st.dataframe(anomalies[['date', 'item_name', 'quantity_sold', 'anomaly', 'event']])

else:
    st.info("👆 Upload your dataset to begin analysis")