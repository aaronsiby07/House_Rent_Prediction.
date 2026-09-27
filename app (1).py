import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Ahmedabad Rent Predictor",
    layout="centered"
)

st.title("Ahmedabad House Rent Predictor")
st.write("Enter the details of a property to estimate its monthly rent.")

# ---------------------------------------------------------
# 1. LOAD THE DATASET
# ---------------------------------------------------------

rent = pd.read_csv("Ahmedabad_rent.csv")
rent = rent.drop_duplicates()

# Clean price
rent["price"] = rent["price"].str.replace(",", "").astype(float)
rent["price"] = np.where(
    rent["price"] < 100,
    rent["price"] * 100000,
    rent["price"]
)

# Clean bathroom
def get_bathroom(x):
    if pd.isnull(x):
        return np.nan
    if "bathroom" in str(x):
        return float(str(x).split()[0])
    return np.nan

rent["bathroom"] = rent["bathroom"].apply(get_bathroom)
rent["bathroom"] = rent["bathroom"].fillna(rent["bathroom"].median())

# ---------------------------------------------------------
# 2. LOAD THE SAVED MODEL AND ENCODERS
# ---------------------------------------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

with open("model_info.pkl", "rb") as file:
    model_info = pickle.load(file)

le_seller = encoders["seller_type"]
le_layout = encoders["layout_type"]
le_property = encoders["property_type"]
le_locality = encoders["locality"]
le_furnish = encoders["furnish_type"]

# The model was trained using these columns in this order.
feature_columns = model_info["feature_columns"]

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
            le_seller.classes_
        )

        layout_choice = st.selectbox(
            "Layout Type",
            le_layout.classes_
        )

        property_choice = st.selectbox(
            "Property Type",
            le_property.classes_
        )

        locality_choice = st.selectbox(
            "Locality",
            le_locality.classes_
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
            le_furnish.classes_
        )

    if st.button("Predict Monthly Rent"):

        # Convert the selected text categories into the
        # same numbers used when the model was trained.
        seller_code = le_seller.transform([seller_choice])[0]
        layout_code = le_layout.transform([layout_choice])[0]
        property_code = le_property.transform([property_choice])[0]
        locality_code = le_locality.transform([locality_choice])[0]
        furnish_code = le_furnish.transform([furnish_choice])[0]

        # Create one row with the user's property details.
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

        # Predict the rent using the saved model.
        prediction = model.predict(input_data)[0]

        # Rent cannot be negative, so display zero if
        # a negative regression result ever occurs.
        prediction = max(0, prediction)

        st.success(f"Estimated Monthly Rent: ₹{prediction:,.0f}")
        st.write(
            "This is an estimated rent based on historical Ahmedabad rental data."
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

    st.write("Number of cleaned listings:", len(rent))
    st.write("Number of localities:", len(le_locality.classes_))
    st.write("MAE:", round(model_info["mae"], 2))
    st.write("RMSE:", round(model_info["rmse"], 2))
    st.write("R²:", round(model_info["r2"], 4))
