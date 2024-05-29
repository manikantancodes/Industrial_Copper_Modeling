import streamlit as st
import pickle


# Load the scaler for the regression model
with open(r"C:\Users\manik\scaling.pkl", "rb") as file:
    scaler_reg = pickle.load(file)

# Load the scaler for the classification model
with open(r"C:\Users\manik\scaling_classify.pkl", "rb") as file:
    scaler_cls = pickle.load(file)

# Load the regression model
with open(r"C:\Users\manik\ExtraTreeRegressor.pkl", "rb") as file:
    et_model = pickle.load(file)

# Load the classification model
with open(r"C:\Users\manik\RandomForestClassification.pkl", "rb") as file:
    rf_model = pickle.load(file)


# Function to map status string to numerical value
def map_status(status_str):
    status_mapping = {"Won": 7.0, "Draft": 0.0, "To be approved": 6.0, "Lost": 1.0, "Not lost for AM": 2.0,
                      "Wonderful": 8.0, "Revised": 5.0, "Offered": 4.0, "Offerable": 3.0}
    return status_mapping.get(status_str, None)

# Function to map item type string to numerical value
def map_item_type(item_type_str):
    item_type_mapping = {"W": 5.0, "WI": 6.0, "S": 3.0, "Others": 1.0, "PL": 2.0, "IPL": 0.0, "SLAWR": 4.0}
    return item_type_mapping.get(item_type_str, None)

# Function to predict selling price using the regression model
def predict_selling_price(quantity, thickness, width, country, status, item_type):
    try:
        # Scale the input features
        x = scaler_reg.transform([[quantity, thickness, width, country, status, item_type, 41.0, 611993, 28]])
        # Make prediction
        predicted_price = et_model.predict(x)
        return predicted_price[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Function to predict status using the classification model
def predict_status(quantity, thickness, width, selling_price, country, item_type):
    try:
        # Scale the input features
        x = scaler_cls.transform([[quantity, thickness, width, selling_price, country, item_type, 10.0, 1670798778, 91]])
        # Make prediction
        predicted_status = rf_model.predict(x)
        if predicted_status == 6:
            return "WON"
        else:
            return "LOST"
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit web app
st.title("ML Prediction App")

quantity = st.number_input("Enter quantity", value=0, step=1)
thickness = st.number_input("Enter thickness", value=0.0, step=0.1, format="%.1f")
width = st.number_input("Enter width", value=0.0, step=0.1, format="%.1f")
country = st.number_input("Enter a country", value=0.0, step=0.1, format="%.1f")
status_str = st.selectbox("Select status", ["Won", "Draft", "To be approved", "Lost", "Not lost for AM", "Wonderful", "Revised", "Offered", "Offerable"])
item_type_str = st.selectbox("Select item type", ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"])

status = map_status(status_str)
item_type = map_item_type(item_type_str)

# Debugging: print inputs to console
st.write(f"Quantity: {quantity}, Thickness: {thickness}, Width: {width}, Country: {country}, Status: {status}, Item Type: {item_type}")

if st.button("Predict"):
    predicted_price = predict_selling_price(quantity, thickness, width, country, status, item_type)
    st.write(f"Predicted Selling Price: {predicted_price}")

    # Check if predicted_price is valid before making status prediction
    if isinstance(predicted_price, float):
        predicted_status = predict_status(quantity, thickness, width, predicted_price, country, item_type)
        st.write(f"Predicted Status: {predicted_status}")
    else:
        st.write(f"Error in predicted price: {predicted_price}")
