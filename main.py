import gradio as gr
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
os.environ["OPENAI_API_KEY"] = "52208f4de55db11441df0ab47c6c30c38244ef683d5bf12a37b028ff198cdbb8"
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

llm = ChatOpenAI(model_name="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.2)
local_tz = pytz.timezone("Asia/Kathmandu")
retriever = None
qa_chain = None

# --- PDF Processing ---
def extract_text_from_pdf(file):
    doc = fitz.open(file.name)
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

# --- Appointment Helpers ---
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
        return "✅ Appointment email sent!"
    except Exception as e:
        return "Not able to send mail for now"


user_info = {"name": "", "phone": "", "email": "", "date": ""}
conversation_state = {
    "awaiting_info": False,
    "current_field": None,
    "info_sequence": ["name", "phone", "email", "date"],
    "asked_fields": [],
}

# --- Chat Logic ---
def chat_with_pdf(message, chat_history):
    global qa_chain, user_info, conversation_state

    def next_missing_field():
        for field in conversation_state["info_sequence"]:
            if not user_info[field]:
                return field
        return None

    if any(kw in message.lower() for kw in ["book", "appointment", "call me", "contact"]):
        conversation_state["awaiting_info"] = True
        conversation_state["asked_fields"] = []
        conversation_state["current_field"] = next_missing_field()

        if conversation_state["current_field"]:
            prompts = {
                "name": " Please provide your name.",
                "phone": " What's your phone number?",
                "email": " What's your email address?",
                "date": " When would you like to schedule the appointment?",
            }
            return prompts[conversation_state["current_field"]]
        else:
            return "ℹ️ All appointment info already provided. Type 'confirm' to proceed."

    if conversation_state["awaiting_info"]:
        field = conversation_state["current_field"]

        if field == "phone" and not is_valid_phone(message):
            return " Invalid phone number. Please enter a valid one (e.g., +9779812345678)."
        if field == "email" and not is_valid_email(message):
            return " Invalid email address. Please try again."

        user_info[field] = message.strip()
        conversation_state["asked_fields"].append(field)
        conversation_state["current_field"] = next_missing_field()

        if conversation_state["current_field"]:
            prompts = {
                "name": " Please provide your name.",
                "phone": " What's your phone number?",
                "email": " What's your email address?",
                "date": " When would you like to schedule the appointment?",
            }
            return prompts[conversation_state["current_field"]]
        else:
            conversation_state["awaiting_info"] = False
            return " Got all your info! Type **'confirm'** to finalize it."

    if "confirm" in message.lower():
        name = user_info["name"]
        phone = user_info["phone"]
        email = user_info["email"]
        date_input = user_info["date"]

        if not (is_valid_phone(phone) and is_valid_email(email)):
            return " Please ensure phone and email are valid before confirming."

        date_parsed = extract_date_llm(date_input)
        if not re.match(r"\d{4}-\d{2}-\d{2}", date_parsed):
            return " Couldn't understand the date. Please try again."

        # result = send_appointment_email(email, name, date_parsed, phone)
       
        user_info.update({"name": "", "phone": "", "email": "", "date": ""})
        conversation_state.update({
            "awaiting_info": False,
            "current_field": None,
            "asked_fields": []
        })
        
        return f" Appointment Confirmed for {date_parsed}"

    if qa_chain:
        response = qa_chain.run(message)
        return response
    else:
        return " Please upload a PDF first."

def handle_pdf_upload(file):
    global retriever, qa_chain
    text = extract_text_from_pdf(file)
    retriever = process_pdf(text)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return " PDF processed. You can now chat with it."

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 💬 PDF Chatbot + Appointment Booking")

    with gr.Row():
        pdf_file = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="Upload Status")

    pdf_file.change(fn=handle_pdf_upload, inputs=pdf_file, outputs=pdf_status)

    chatbot = gr.ChatInterface(fn=chat_with_pdf, chatbot=gr.Chatbot(label="DocumentAnswerer"))

if __name__ == "__main__":
    demo.launch()
