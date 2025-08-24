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
                        # Create a prominent download section
                        st.markdown("---")
                        st.subheader("Download Your Translated Document")
                        
                        # Construct proper download URL
                        if translated_file.startswith('/'):
                            download_url = DOWNLOAD_API + translated_file
                        else:
                            download_url = f"{DOWNLOAD_API}/{translated_file}"
                        
                        # Create columns for better layout
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            # Try to fetch the file and provide direct download
                            try:
                                file_response = requests.get(download_url)
                                if file_response.status_code == 200:
                                    # Determine file extension from the translated_file name
                                    file_extension = translated_file.split('.')[-1] if '.' in translated_file else 'pdf'
                                    mime_type = 'application/pdf' if file_extension == 'pdf' else 'text/plain'
                                    
                                    st.download_button(
                                        label="Download Translated File",
                                        data=file_response.content,
                                        file_name=f"translated_{uploaded_file.name}",
                                        mime=mime_type,
                                        type="primary"
                                    )
                                else:
                                    # Fallback to link if direct download fails
                                    st.markdown(f"[Download Translated File]({download_url})")
                            except Exception as download_error:
                                # Fallback to link if there's any error
                                st.markdown(f"[Download Translated File]({download_url})")
                                st.caption("Click the link above to download your translated document")
                        
                        st.markdown("---")
                else:
                    st.error(result.get("message", "Unknown error"))
            else:
                st.error(f"❌ API Error: {response.status_code}")
        except Exception as e:
            st.error(f"🚨 Connection Error: {e}")
