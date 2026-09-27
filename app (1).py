import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ahmedabad Rent Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("Ahmedabad House Rent Predictor")
st.write("Enter the details of a property to estimate its monthly rent.")


# ---------------------------------------------------------
# 1. LOAD AND PREPARE THE DATASET
# ---------------------------------------------------------

@st.cache_data
def load_data():

    rent = pd.read_csv("Ahmedabad_rent.csv")
    rent = rent.drop_duplicates().copy()

    # Clean price safely whether it is stored as text or numeric
    rent["price"] = (
        rent["price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )
    rent["price"] = pd.to_numeric(rent["price"], errors="coerce")

    # If prices below 100 represent lakhs, convert them to rupees
    rent.loc[rent["price"] < 100, "price"] = (
        rent.loc[rent["price"] < 100, "price"] * 100000
    )

    # Convert numeric columns safely
    for col in ["bedroom", "area"]:
        rent[col] = pd.to_numeric(rent[col], errors="coerce")

    # Clean bathroom safely
    def clean_bathroom(x):
        if pd.isna(x):
            return np.nan

        text = str(x).strip().lower()

        try:
            return float(text.split()[0])
        except (ValueError, IndexError):
            return np.nan

    rent["bathroom"] = rent["bathroom"].apply(clean_bathroom)

    # Fill numeric missing values
    rent["bedroom"] = rent["bedroom"].fillna(rent["bedroom"].median())
    rent["area"] = rent["area"].fillna(rent["area"].median())
    rent["bathroom"] = rent["bathroom"].fillna(rent["bathroom"].median())

    # Required categorical columns
    categorical_cols = [
        "seller_type",
        "layout_type",
        "property_type",
        "locality",
        "furnish_type"
    ]

    # Fill missing categorical values
    for col in categorical_cols:
        rent[col] = rent[col].fillna(
            rent[col].mode()[0]
        ).astype(str)

    # Remove rows where target is unavailable
    rent = rent.dropna(subset=["price"])

    return rent


rent = load_data()


# ---------------------------------------------------------
# 2. TRAIN LINEAR REGRESSION MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_model(data):

    df = data.copy()

    # Create encoders
    encoders = {}

    categorical_cols = [
        "seller_type",
        "layout_type",
        "property_type",
        "locality",
        "furnish_type"
    ]

    # Label encode categorical features
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    # Features used by the model
    feature_columns = [
        "seller_type",
        "bedroom",
        "layout_type",
        "property_type",
        "locality",
        "area",
        "furnish_type",
        "bathroom"
    ]

    X = df[feature_columns]
    y = df["price"]

    # Same 80/20 split used for the project
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return (
        model,
        encoders,
        feature_columns,
        mae,
        rmse,
        r2
    )


(
    model,
    encoders,
    feature_columns,
    mae,
    rmse,
    r2
) = train_model(rent)


# ---------------------------------------------------------
# 3. SIDEBAR MENU
# ---------------------------------------------------------

page = st.sidebar.radio(
    "Choose a page",
    ["Rent Prediction", "Project Information"]
)


# ---------------------------------------------------------
# 4. RENT PREDICTION PAGE
# ---------------------------------------------------------

if page == "Rent Prediction":

    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:

        seller_choice = st.selectbox(
            "Seller Type",
            encoders["seller_type"].classes_
        )

        layout_choice = st.selectbox(
            "Layout Type",
            encoders["layout_type"].classes_
        )

        property_choice = st.selectbox(
            "Property Type",
            encoders["property_type"].classes_
        )

        locality_choice = st.selectbox(
            "Locality",
            encoders["locality"].classes_
        )

    with col2:

        bedroom = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        area = st.number_input(
            "Area (sq.ft)",
            min_value=100,
            max_value=100000,
            value=1200,
            step=50
        )

        bathroom = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=15,
            value=2,
            step=1
        )

        furnish_choice = st.selectbox(
            "Furnishing Type",
            encoders["furnish_type"].classes_
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button("Predict Monthly Rent", type="primary"):

        # Encode selected categories using the same
        # encoders used during model training
        seller_code = encoders["seller_type"].transform(
            [seller_choice]
        )[0]

        layout_code = encoders["layout_type"].transform(
            [layout_choice]
        )[0]

        property_code = encoders["property_type"].transform(
            [property_choice]
        )[0]

        locality_code = encoders["locality"].transform(
            [locality_choice]
        )[0]

        furnish_code = encoders["furnish_type"].transform(
            [furnish_choice]
        )[0]

        # Create prediction row in exactly the same
        # feature order used during training
        input_data = pd.DataFrame(
            [[
                seller_code,
                bedroom,
                layout_code,
                property_code,
                locality_code,
                area,
                furnish_code,
                bathroom
            ]],
            columns=feature_columns
        )

        prediction = model.predict(input_data)[0]

        # Prevent negative rent
        prediction = max(0, prediction)

        st.success(
            f"Estimated Monthly Rent: ₹{prediction:,.0f}"
        )

        st.write(
            "This is an estimated rent based on historical "
            "Ahmedabad rental data."
        )


# ---------------------------------------------------------
# 5. PROJECT INFORMATION PAGE
# ---------------------------------------------------------

else:

    st.subheader("Project Information")

    st.write(
        "This project uses Ahmedabad rental listings to estimate "
        "the monthly rent of a property using Linear Regression."
    )

    st.write(
        "The model uses property and rental characteristics such "
        "as seller type, bedroom count, layout type, property type, "
        "locality, area, furnishing type, and bathroom count."
    )

    st.divider()

    st.write("### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Cleaned Listings",
            f"{len(rent):,}"
        )

    with col2:
        st.metric(
            "Localities",
            f"{len(encoders['locality'].classes_):,}"
        )

    st.write("### Model Performance")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "MAE",
            f"₹{mae:,.2f}"
        )

    with m2:
        st.metric(
            "RMSE",
            f"₹{rmse:,.2f}"
        )

    with m3:
        st.metric(
            "R² Score",
            f"{r2 * 100:.2f}%"
        )
