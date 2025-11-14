import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

def load_data():
    df = pd.read_excel("superstore.xlsx")

    remove_cols = [
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'Country', 'Postal Code'
    ]
    df = df.drop(columns=[c for c in remove_cols if c in df.columns], errors="ignore")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="ignore")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="ignore")

    return df

df = load_data()

st.sidebar.header("Filter Options")

regions = df["Region"].dropna().unique().tolist()
categories = df["Category"].dropna().unique().tolist()

selected_regions = st.sidebar.multiselect("Select Region", regions, default=regions)
selected_categories = st.sidebar.multiselect("Select Category", categories, default=categories)

filtered_df = df[
    (df["Region"].isin(selected_regions)) &
    (df["Category"].isin(selected_categories))
]

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()
total_orders = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Avg Discount", f"{avg_discount:.2%}")
col4.metric("Total Orders", f"{total_orders:,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Sales Overview", "Profit Trend", "Prediction"])

with tab1:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Sales by Sub-Category")
        fig, ax = plt.subplots()
        sub_sales = filtered_df.groupby("Sub-Category")["Sales"].sum().sort_values()
        ax.bar(sub_sales.index, sub_sales.values)
        ax.set_ylabel("Sales")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with colB:
        st.subheader("Sales by Region")
        region_sales = filtered_df.groupby("Region")["Sales"].sum()
        fig, ax = plt.subplots()
        ax.pie(region_sales.values, labels=region_sales.index, autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

with tab2:
    st.subheader("Profit Over Time")

    daily_profit = filtered_df.groupby("Order Date")["Profit"].sum()
    fig, ax = plt.subplots()
    ax.plot(daily_profit.index, daily_profit.values)
    ax.set_ylabel("Profit")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    st.subheader("Predict Profit (Random Forest)")

    X = df[["Sales", "Discount", "Quantity"]]
    y = df["Profit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    col1, col2, col3 = st.columns(3)
    inp_sales = col1.number_input("Sales", min_value=0.0, value=500.0)
    inp_discount = col2.number_input("Discount", min_value=0.0, max_value=1.0, value=0.1)
    inp_qty = col3.number_input("Quantity", min_value=1, value=2)

    if st.button("Predict Profit"):
        pred = model.predict([[inp_sales, inp_discount, inp_qty]])[0]
        st.success(f"Predicted Profit: ${pred:,.2f}")
