
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("Sales Forecasting & Demand Intelligence System")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

if page == "Sales Overview":
    st.header("Sales Overview Dashboard")

    # Load Dataset
    df = pd.read_csv("train.csv", encoding="latin1")

    # Convert Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    # Create Year Column
    df["Year"] = df["Order Date"].dt.year

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Select Region",
            ["All"] + sorted(df["Region"].unique())
        )

    with col2:
        category = st.selectbox(
            "Select Category",
            ["All"] + sorted(df["Category"].unique())
        )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == region]

    if category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == category]

    # Total Sales by Year
    st.subheader("Total Sales by Year")

    yearly_sales = filtered_df.groupby("Year")["Sales"].sum()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(yearly_sales.index.astype(str), yearly_sales.values)
    st.pyplot(fig)

    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")

    monthly_sales = filtered_df.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"].sum()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_sales.index, monthly_sales.values, marker="o")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Summary
    st.subheader("Summary")

    st.metric("Total Sales", f"{filtered_df['Sales'].sum():,.2f}")

elif page == "Forecast Explorer":
    st.header("Forecast Explorer")

    # Load Dataset
    df = pd.read_csv("train.csv", encoding="latin1")

    # Convert Order Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    # Select Forecast Type
    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    # Select Category or Region
    if forecast_type == "Category":
        selected = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )
        data = df[df["Category"] == selected]

    else:
        selected = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )
        data = df[df["Region"] == selected]

    # Forecast Horizon
    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # Monthly Sales
    monthly_sales = data.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"].sum()

    # Simple Forecast (Last Value Repeated)
    forecast = [monthly_sales.iloc[-1]] * horizon

    future_dates = pd.date_range(
        monthly_sales.index[-1] + pd.offsets.MonthEnd(),
        periods=horizon,
        freq="ME"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly_sales.index,
        monthly_sales.values,
        label="Actual Sales",
        marker="o"
    )

    ax.plot(
        future_dates,
        forecast,
        label="Forecast",
        marker="o",
        linestyle="--"
    )

    ax.set_title("Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    # Forecast Table
    forecast_df = pd.DataFrame({
        "Forecast Month": future_dates.strftime("%Y-%m"),
        "Predicted Sales": forecast
    })

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

    # Dummy Metrics (Replace with your Task 3 model results)
    mae = 120.45
    rmse = 180.20

    col1, col2 = st.columns(2)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

elif page == "Anomaly Report":
    st.header("Anomaly Report")

    from sklearn.ensemble import IsolationForest
    import numpy as np

    # Load Dataset
    df = pd.read_csv("train.csv", encoding="latin1")

    # Convert Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    # Weekly Sales
    weekly_sales = df.groupby(
        pd.Grouper(key="Order Date", freq="W")
    )["Sales"].sum().reset_index()

    # Isolation Forest Model
    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = model.fit_predict(
        weekly_sales[["Sales"]]
    )

    # Convert labels
    weekly_sales["Anomaly"] = weekly_sales["Anomaly"].map(
        {1: "Normal", -1: "Anomaly"}
    )

    # -------------------------------
    # Plot
    # -------------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    normal = weekly_sales[
        weekly_sales["Anomaly"] == "Normal"
    ]

    anomaly = weekly_sales[
        weekly_sales["Anomaly"] == "Anomaly"
    ]

    ax.plot(
        normal["Order Date"],
        normal["Sales"],
        label="Normal Sales"
    )

    ax.scatter(
        anomaly["Order Date"],
        anomaly["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_title("Weekly Sales with Detected Anomalies")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # Anomaly Table
    # -------------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly[
            ["Order Date", "Sales", "Anomaly"]
        ].reset_index(drop=True)
    )

    # -------------------------------
    # Summary
    # -------------------------------

    st.metric(
        "Total Anomalies",
        len(anomaly)
    )

elif page == "Demand Segments":
    st.header("Product Demand Segments")

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Load Dataset
    df = pd.read_csv("train.csv", encoding="latin1")

    # Convert Order Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    # Create Year Column
    df["Year"] = df["Order Date"].dt.year

    # ---------------------------------------
    # Aggregate Data by Sub-Category
    # ---------------------------------------

    product_data = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Average_Sales=("Sales", "mean"),
        Total_Orders=("Order ID", "count")
    ).reset_index()

    # ---------------------------------------
    # Standardize Features
    # ---------------------------------------

    features = product_data[
        ["Total_Sales", "Average_Sales", "Total_Orders"]
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # ---------------------------------------
    # KMeans Clustering
    # ---------------------------------------

    kmeans = KMeans(n_clusters=4, random_state=42)

    product_data["Cluster"] = kmeans.fit_predict(scaled_features)

    # ---------------------------------------
    # Demand Labels
    # ---------------------------------------

    demand_labels = {
        0: "High Volume",
        1: "Growing Demand",
        2: "Stable Demand",
        3: "Low Demand"
    }

    product_data["Demand Group"] = product_data["Cluster"].map(demand_labels)

    # ---------------------------------------
    # PCA
    # ---------------------------------------

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(scaled_features)

    product_data["PCA1"] = pca_result[:, 0]
    product_data["PCA2"] = pca_result[:, 1]

    # ---------------------------------------
    # Scatter Plot
    # ---------------------------------------

    st.subheader("Product Demand Clusters")

    fig, ax = plt.subplots(figsize=(10,6))

    for cluster in sorted(product_data["Cluster"].unique()):

        cluster_data = product_data[
            product_data["Cluster"] == cluster
        ]

        ax.scatter(
            cluster_data["PCA1"],
            cluster_data["PCA2"],
            label=demand_labels[cluster],
            s=100
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Demand Segmentation using K-Means")
    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------
    # Cluster Table
    # ---------------------------------------

    st.subheader("Demand Segment Table")

    st.dataframe(
        product_data[
            [
                "Sub-Category",
                "Cluster",
                "Demand Group",
                "Total_Sales",
                "Average_Sales",
                "Total_Orders"
            ]
        ]
    )

    # ---------------------------------------
    # Cluster Summary
    # ---------------------------------------

    st.subheader("Cluster Summary")

    st.write(product_data["Demand Group"].value_counts())
