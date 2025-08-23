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
    "Portuguese": "pt_XX",  
    "Dutch": "nl_XX",
    "Polish": "pl_PL",      
    "Romanian": "ro_RO",    
    "Russian": "ru_RU",     
    "Ukrainian": "uk_UK",   
    "Bulgarian": "bg_BG",   
    "Czech": "cs_CZ",       
    "Danish": "da_DK",      
    "Finnish": "fi_FI",     
    "Greek": "el_EL",       
    "Hungarian": "hu_HU",   
    "Latvian": "lv_LV",     
    "Lithuanian": "lt_LT",  
    "Slovak": "sk_SK",      
    "Slovenian": "sl_SI",   
    "Swedish": "sv_SE",     
    "Croatian": "hr_HR",    
    "Serbian": "sr_XX",     
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
        data = {"src_lang": languages[source_lang], "tgt_lang": languages[target_lang]}

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
