import streamlit as st
import requests

UPLOAD_API = "http://localhost:5000/process"
DOWNLOAD_API = "http://localhost:5000/download-final-pdf"


languages = {
    "English": "en_XX",
    "French": "fr_XX",
    "German": "de_DE",   
    "Spanish": "es_XX",
    "Italian": "it_IT",  
    "Portuguese (European)": "por_XX", 
    "Dutch": "nl_XX",    
    "Polish": "pl_XX",   
    "Romanian": "ro_XX", 
    "Russian": "ru_XX",  
    "Ukrainian": "uk_XX",
    "Bulgarian": "bg_XX",
    "Czech": "cs_XX",    
    "Danish": "da_XX",   
    "Finnish": "fi_XX",  
    "Greek": "el_XX",    
    "Hungarian": "hu_XX",
    "Latvian": "lv_XX",  
    "Lithuanian": "lt_XX",
    "Norwegian": "no_XX", 
    "Slovak": "sk_XX",   
    "Slovenian": "sl_XX",
    "Swedish": "sv_XX",  
    "Croatian": "hr_XX", 
    "Serbian (Latin)": "sr_XX", 
    "Arabic": "ar_AR"
}


st.title("Visa Document Translator – Upload & Download Translations")

source_lang = st.selectbox("Select Original Language:", list(languages.keys()))
target_lang = st.selectbox("Select Translation Language:", list(languages.keys()))

uploaded_file = st.file_uploader("Upload PDF or Image", type=["txt","pdf","png","jpg","jpeg"])

if uploaded_file is not None:
    if st.button("Upload & Translate"):
        # Prepare form data for your /upload API
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"source_lang": languages[source_lang], "target_lang": languages[target_lang]}

        try:
            response = requests.post(UPLOAD_API, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    st.success("✅ Translation Complete!")
                    translated_file = result.get("text_file")
                    if translated_file:
                        download_url = DOWNLOAD_API + translated_file
                        st.markdown(f"[⬇️ Download Translated File]({download_url})", unsafe_allow_html=True)
                else:
                    st.error(result.get("message", "Unknown error"))
            else:
                st.error(f"❌ API Error: {response.status_code}")
        except Exception as e:
            st.error(f"🚨 Connection Error: {e}")
