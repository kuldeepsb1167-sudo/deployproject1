
import streamlit as st
from google import genai
from pypdf import PdfReader
from io import BytesIO

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(
    page_title="Gemini Document Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Summarizer using Gemini")
st.write("Upload a PDF or TXT document and generate a summary using Gemini.")

# -------------------------------
# Load API key from Streamlit Secrets
# -------------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]

    if not api_key:
        st.error("GEMINI_API_KEY is empty in Streamlit Secrets.")
        st.stop()

    client = genai.Client(api_key=api_key)

except Exception as e:
    st.error(
        "Could not initialize Gemini. "
        "Check your Streamlit Secrets and installed packages."
    )
    st.code(str(e))
    st.stop()

# -------------------------------
# Upload document
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    try:
        # Extract text from PDF or TXT
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))

            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")

            document_text = "\n".join(pages)

        else:
            document_text = uploaded_file.getvalue().decode(
                "utf-8-sig",
                errors="replace"
            )

        if not document_text.strip():
            st.warning(
                "No readable text was found. "
                "If this is a scanned PDF, OCR may be required."
            )
            st.stop()

        st.success("Document uploaded and text extracted successfully.")

        with st.expander("View extracted document text"):
            st.text_area(
                "Extracted text",
                document_text,
                height=250
            )

        # -------------------------------
        # Summarize document
        # -------------------------------
        if st.button("✨ Summarize Document", type="primary"):

            prompt = f"""
You are an expert document analyst.

Analyze the document below and provide:

1. Short summary
2. Key points
3. Important facts and figures
4. Main conclusion

Use clear headings and bullet points.
Do not invent facts that are not present in the document.

DOCUMENT:
{document_text}
"""

            with st.spinner("Gemini is summarizing your document..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt
                    )

                    if response.text:
                        st.subheader("📝 Gemini Summary")
                        st.markdown(response.text)
                    else:
                        st.warning(
                            "Gemini returned an empty response. "
                            "Try another document."
                        )

                except Exception as e:
                    st.error("Gemini could not summarize the document.")
                    st.code(str(e))

    except Exception as e:
        st.error("Could not read the uploaded document.")
        st.code(str(e))