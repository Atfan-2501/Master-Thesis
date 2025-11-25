import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from firebase_config import initialize_firebase
import time

# Initialize Firebase
firestore_client = initialize_firebase()

def fetch_data(collection_name):
    try:
        docs = firestore_client.collection(collection_name).stream()
        data = [doc.to_dict() for doc in docs]
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def main():
    # Custom styles
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
            }
            .main .block-container {
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            .stButton>button {
                background: linear-gradient(90deg, #ff7e5f, #feb47b);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                transition: all 0.3s ease-in-out;
            }
            .stButton>button:hover {
                transform: scale(1.05);
                background: linear-gradient(90deg, #ff6a6a, #ffb199);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("✨ Firestore Data Viewer ✨")
    
    # Get all collection names
    collections = firestore_client.collections()
    collection_names = [col.id for col in collections]
    
    if not collection_names:
        st.warning("No collections found in Firestore.")
        return
    
    # Collection selection dropdown with animation
    selected_collection = st.selectbox("📂 Select a collection", collection_names)
    
    # Fetch data from selected collection
    df = fetch_data(selected_collection)
    
    if df.empty:
        st.warning("⚠ No data available in this collection.")
        return
    
    # Display table with a border effect
    st.markdown("**📊 Data Preview:**")
    st.dataframe(df, use_container_width=True)
    
    # Download button with animation
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        if st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_collection}.csv",
            mime="text/csv"
        ):
            st.success("✅ Download started!")
            time.sleep(1)
            st.toast("🎉 File downloaded successfully!")

if __name__ == "__main__":
    main()
