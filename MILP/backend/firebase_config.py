import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

# Initialize Firebase using Streamlit secrets
def initialize_firebase():
    if not firebase_admin._apps:
        firebase_credentials = {
            "type": st.secrets["firebase_credentials"]["type"],
            "project_id": st.secrets["firebase_credentials"]["project_id"],
            "private_key_id": st.secrets["firebase_credentials"]["private_key_id"],
            "private_key": st.secrets["firebase_credentials"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["firebase_credentials"]["client_email"],
            "client_id": st.secrets["firebase_credentials"]["client_id"],
            "auth_uri": st.secrets["firebase_credentials"]["auth_uri"],
            "token_uri": st.secrets["firebase_credentials"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase_credentials"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase_credentials"]["client_x509_cert_url"]
        }
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)

    return firestore.client()
