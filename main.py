# app.py

import streamlit as st
import fitz
import os
import re
from datetime import datetime
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA

# --- Setup ---
os.environ["OPENAI_API_KEY"] = "your_together_api_key"
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

llm = ChatOpenAI(model_name="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.2)
local_tz = pytz.timezone("Asia/Kathmandu")

# --- PDF Processing ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_type="similarity", k=4)

# --- Appointment Logic ---
def extract_date_llm(user_input):
    today_local = datetime.now(local_tz).strftime('%Y-%m-%d')
    prompt = f"""Today is {today_local}. Convert this expression to an ISO date format (YYYY-MM-DD): '{user_input}'. Only return the date. Do not explain."""
    response = llm.invoke(prompt)
    return response.content.strip()

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^\+?\d{10}$", phone)

def send_appointment_email(recipient_email, name, date, phone):
    sender_email = "your_email@gmail.com"
    app_password = "your_app_password"

    subject = "Your Appointment Confirmation"
    body = f"""Hello {name},\n\nThis is a confirmation for your appointment on {date}.\nWe'll reach out to you at {phone} if needed.\n\nThank you,\nPDF Chatbot"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        st.success("✅ Appointment email sent!")
    except Exception as e:
        st.error("❌ Failed to send email: " + str(e))

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📄 Chat with Your PDF")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.retriever is None:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        retriever = process_pdf(raw_text)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        st.session_state.retriever = retriever
        st.session_state.qa_chain = qa_chain
    st.success("✅ PDF is ready! Ask your questions below.")

if st.session_state.qa_chain:
    user_input = st.text_input("Ask something or type 'book appointment'")

    if user_input:
        if any(kw in user_input.lower() for kw in ["book", "appointment", "call me", "contact"]):
            with st.form("appointment_form"):
                name = st.text_input("👤 Your Name")
                phone = st.text_input("📞 Phone Number")
                email = st.text_input("📧 Email Address")
                date_input = st.text_input("📅 Preferred Date (e.g., next Monday)")

                submitted = st.form_submit_button("Book Appointment")
                if submitted:
                    if not is_valid_phone(phone):
                        st.warning("⚠️ Invalid phone number")
                    elif not is_valid_email(email):
                        st.warning("⚠️ Invalid email address")
                    else:
                        date_parsed = extract_date_llm(date_input)
                        if not re.match(r"\d{4}-\d{2}-\d{2}", date_parsed):
                            st.warning("⚠️ Couldn't understand the date")
                        else:
                            st.success(f"✅ Appointment Confirmed for {date_parsed}")
                            send_appointment_email(email, name, date_parsed, phone)
        else:
            response = st.session_state.qa_chain.run(user_input)
            st.write("🤖", response)
