import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os # For checking file existence

# --- Configuration and File Loading ---
st.set_page_config(layout="wide", page_title="Shopper Spectrum Analytics")

# Define file paths for models and data
MODEL_PATH = 'kmeans_model.joblib'
SCALER_PATH = 'scaler.joblib'
CLUSTER_MAP_PATH = 'cluster_segment_map.joblib'
ITEM_SIM_PATH = 'item_similarity_df.joblib'
PRODUCT_DESC_PATH = 'all_product_descriptions.joblib'

# Function to load models and data safely
@st.cache_resource # Cache the models to avoid re-loading on every rerun
def load_resources():
    resources = {}
    try:
        resources['kmeans_model'] = joblib.load(MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['cluster_segment_map'] = joblib.load(CLUSTER_MAP_PATH)
        resources['item_similarity_df'] = joblib.load(ITEM_SIM_PATH)
        resources['all_product_descriptions'] = joblib.load(PRODUCT_DESC_PATH)
        st.success("All models and data loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e}. Please ensure all .joblib files are in the same directory as this app.py.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()
    return resources

# Load all resources at the start
resources = load_resources()
kmeans_model = resources['kmeans_model']
scaler = resources['scaler']
cluster_segment_map = resources['cluster_segment_map']
item_similarity_df = resources['item_similarity_df']
all_product_descriptions = resources['all_product_descriptions']

# --- Helper Functions ---

def get_recommendations(product_description, n=5):
    """
    Returns the top N most similar products to a given product description.
    """
    if product_description not in item_similarity_df.columns:
        return []

    product_similarities = item_similarity_df[product_description]
    similar_products = product_similarities.sort_values(ascending=False).index.tolist()
    
    # Remove the product itself from recommendations
    if product_description in similar_products:
        similar_products.remove(product_description) 
    
    return similar_products[:n]

def predict_customer_segment(recency, frequency, monetary):
    """
    Predicts the customer segment based on RFM values.
    """
    # Create a DataFrame for the input
    customer_data = pd.DataFrame([[recency, frequency, monetary]],
                                 columns=['Recency', 'Frequency', 'Monetary'])

    # Apply log transformation (same as during training)
    customer_data_log = customer_data.apply(np.log1p)

    # Scale the data using the loaded scaler
    customer_scaled = scaler.transform(customer_data_log)

    # Predict the cluster
    predicted_cluster = kmeans_model.predict(customer_scaled)[0]

    # Map the cluster to segment name
    segment_name = cluster_segment_map.get(predicted_cluster, "Unknown Segment")
    return segment_name

# --- Streamlit UI ---

st.title("🛍️ Shopper Spectrum: Customer Analytics")
st.markdown("Unlock insights into customer behavior and personalize product recommendations.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Customer Segmentation", "✨ Product Recommendations", "ℹ️ About Project"])

with tab1:
    st.header("Customer Segmentation (RFM Analysis)")
    st.write("Enter Recency (days since last purchase), Frequency (number of purchases), and Monetary (total spend) to predict customer segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency_input = st.number_input("Recency (Days Ago)", min_value=0, max_value=730, value=30, help="Number of days since the customer's last purchase.")
    with col2:
        frequency_input = st.number_input("Frequency (Purchases)", min_value=1, max_value=100, value=5, help="Total number of unique purchases made by the customer.")
    with col3:
        monetary_input = st.number_input("Monetary (Total Spend £)", min_value=0.0, max_value=50000.0, value=500.0, help="Total amount of money spent by the customer.")

    if st.button("Predict Customer Segment"):
        if recency_input is not None and frequency_input is not None and monetary_input is not None:
            if monetary_input == 0: # Handle log1p(0) for monetary during prediction as well.
                st.warning("Monetary value cannot be zero for meaningful segmentation. Please enter a value greater than 0 if the customer has spent money.")
            else:
                segment = predict_customer_segment(recency_input, frequency_input, monetary_input)
                st.markdown(f"**Predicted Customer Segment: <span style='color:green; font-size:24px;'>{segment}</span>**", unsafe_allow_html=True)
                if segment == 'High-Value':
                    st.info("This customer is a top-tier customer. Consider loyalty programs or exclusive offers.")
                elif segment == 'Regular':
                    st.info("This customer makes consistent purchases. Encourage repeat business with tailored promotions.")
                elif segment == 'Occasional':
                    st.info("This customer purchases infrequently. Try re-engagement campaigns or special discounts to encourage more purchases.")
                elif segment == 'At-Risk':
                    st.info("This customer has not purchased recently and/or has low activity. Implement retention strategies immediately.")
                else:
                    st.info("Segment details not available.")
        else:
            st.warning("Please enter valid numbers for all RFM fields.")


with tab2:
    st.header("Product Recommendations")
    st.write("Select a product to find similar items based on customer purchasing patterns.")

    # Dropdown for product selection
    selected_product = st.selectbox(
        "Select a product:",
        options=[""] + sorted(list(item_similarity_df.columns)), # Add an empty option and sort for usability
        index=0, # Default to the empty option
        help="Start typing to search for a product or select from the list."
    )

    if st.button("Get Recommendations"):
        if selected_product and selected_product != "":
            recommendations = get_recommendations(selected_product, n=5)
            if recommendations:
                st.subheader(f"Top 5 Similar Products to '{selected_product}':")
                for i, product in enumerate(recommendations):
                    st.write(f"{i+1}. {product}")
            else:
                st.info(f"Could not find recommendations for '{selected_product}'. It might be a unique item or not enough co-purchase data.")
        else:
            st.warning("Please select a product from the list.")

with tab3:
    st.header("About This Project")
    st.markdown("""
    This project, **"Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce"**,
    aims to analyze e-commerce transaction data to understand customer purchasing behaviors,
    segment customers, and recommend relevant products.

    **Key Objectives:**
    * Uncover patterns in customer purchase behavior.
    * Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
    * Develop a product recommendation system using collaborative filtering.

    **Technologies Used:**
    * **Python:** Core programming language.
    * **Pandas & NumPy:** Data manipulation and numerical operations.
    * **Scikit-learn:** Machine learning algorithms (KMeans for clustering, Cosine Similarity).
    * **Matplotlib & Seaborn:** Data visualization.
    * **Streamlit:** For building interactive web applications.

    **Customer Segments:**
    * **High-Value:** Customers with low Recency, high Frequency, and high Monetary values.
    * **Regular:** Customers with medium Recency, Frequency, and Monetary values.
    * **Occasional:** Customers with low Frequency and Monetary values, but not necessarily very high Recency (may still be relatively recent).
    * **At-Risk:** Customers with high Recency and low Frequency and Monetary values, indicating they haven't purchased in a long time and are not frequent/high spenders.

    This application demonstrates the power of data analytics and machine learning in enhancing
    e-commerce strategies and customer engagement.
    """)
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-light.svg", width=150) # Example logo