
import os
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BizzBudy AI",
    page_icon="🛍️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PRODUCTS_FILE = BASE_DIR / "products.csv"
FAQS_FILE = BASE_DIR / "faqs.json"
ESCALATIONS_FILE = DATA_DIR / "escalations.csv"

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# 2. LOAD API KEY SECURELY
# --------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        API_KEY = None

client = None

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as error:
        print(f"Gemini client initialization error: {error}")


# --------------------------------------------------
# 3. LOAD PRODUCTS
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "product_id",
    "name",
    "category",
    "price",
    "stock",
    "description",
]


@st.cache_data
def load_products():
    if not PRODUCTS_FILE.exists():
        raise FileNotFoundError(
            "products.csv was not found beside app.py."
        )

    df = pd.read_csv(PRODUCTS_FILE)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing products.csv columns: {', '.join(sorted(missing))}"
        )

    df["name"] = df["name"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["product_id"] = df["product_id"].fillna("").astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0)

    df = df.dropna(subset=["price"])
    return df


# --------------------------------------------------
# 4. LOAD FAQS
# --------------------------------------------------

@st.cache_data
def load_faqs():
    if not FAQS_FILE.exists():
        raise FileNotFoundError(
            "faqs.json was not found beside app.py."
        )

    with open(FAQS_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

    if not isinstance(faqs, list):
        raise ValueError("faqs.json must contain a JSON list.")

    for faq in faqs:
        if not isinstance(faq, dict):
            raise ValueError("Each FAQ must be a JSON object.")
        if "question" not in faq or "answer" not in faq:
            raise ValueError(
                "Each FAQ must contain question and answer fields."
            )

    return faqs


# --------------------------------------------------
# 5. LOAD BUSINESS DATA
# --------------------------------------------------

try:
    products = load_products()
    faqs = load_faqs()
    data_error = None
except Exception as error:
    products = pd.DataFrame(columns=REQUIRED_COLUMNS)
    faqs = []
    data_error = str(error)


# --------------------------------------------------
# 6. SEARCH HELPERS
# --------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "i", "you",
    "we", "our", "your", "what", "whats", "how", "much",
    "many", "do", "does", "can", "could", "please", "tell",
    "me", "about", "for", "of", "to", "in", "on", "and",
    "it", "this", "that", "my", "with", "have", "has",
    "price", "cost", "buy", "purchase", "available",
}


def tokens(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {word for word in words if word not in STOP_WORDS}


def search_products(question, limit=5):
    question_words = tokens(question)

    if not question_words or products.empty:
        return []

    matches = []

    for _, row in products.iterrows():
        searchable = (
            f"{row['product_id']} {row['name']} "
            f"{row['category']} {row['description']}"
        )
        product_words = tokens(searchable)
        overlap = question_words & product_words

        if overlap:
            matches.append((len(overlap), row.to_dict()))

    matches.sort(key=lambda item: item[0], reverse=True)

    return [row for _, row in matches[:limit]]


def search_faqs(question, limit=3):
    question_words = tokens(question)
    scored = []

    for faq in faqs:
        faq_words = tokens(faq["question"])
        overlap = question_words & faq_words

        if overlap:
            scored.append((len(overlap), faq))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [faq for _, faq in scored[:limit]]


def is_price_question(question):
    return bool(
        re.search(
            r"\b(price|cost|how much|rate|priced)\b",
            question.lower(),
        )
    )


def is_stock_question(question):
    return bool(
        re.search(
            r"\b(in stock|available|availability|left|stock)\b",
            question.lower(),
        )
    )


def money(value):
    return f"₹{float(value):,.2f}".removesuffix(".00")


# --------------------------------------------------
# 7. HUMAN SUPPORT ESCALATION
# --------------------------------------------------

ESCALATION_KEYWORDS = [
    "complaint",
    "angry",
    "fraud",
    "scam",
    "wrong product",
    "damaged",
    "broken",
    "refund",
    "chargeback",
    "legal action",
    "speak to human",
    "talk to human",
    "human agent",
    "human support",
    "manager",
]


def requires_human_review(question):
    question_lower = question.lower()
    return any(
        phrase in question_lower
        for phrase in ESCALATION_KEYWORDS
    )


def save_escalation(question, reason):
    record = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "reason": reason,
    }])

    if ESCALATIONS_FILE.exists():
        record.to_csv(
            ESCALATIONS_FILE,
            mode="a",
            header=False,
            index=False,
        )
    else:
        record.to_csv(
            ESCALATIONS_FILE,
            index=False,
        )


# --------------------------------------------------
# 8. FORMAT PRODUCT DETAILS
# --------------------------------------------------

def format_product(product):
    stock = int(product["stock"])

    if stock <= 0:
        stock_text = "Currently out of stock"
    else:
        stock_text = f"{stock} in stock"

    return (
        f"**{product['name']}**\n\n"
        f"- Product ID: {product['product_id']}\n"
        f"- Price: {money(product['price'])}\n"
        f"- Availability: {stock_text}\n"
        f"- Description: {product['description']}"
    )


# --------------------------------------------------
# 9. GENERATE A CUSTOMER ANSWER
# --------------------------------------------------

def answer_customer(question):
    question = question.strip()

    if not question:
        return {
            "answer": "Please type a question.",
            "escalate": False,
            "reason": "",
        }

    if data_error:
        return {
            "answer": (
                "Business data could not be loaded. "
                "Please check products.csv and faqs.json."
            ),
            "escalate": True,
            "reason": "Business data loading error",
        }

    # Escalate sensitive customer-service cases.
    if requires_human_review(question):
        try:
            save_escalation(question, "Human review requested")
        except OSError as error:
            print(f"Escalation save error: {error}")

        return {
            "answer": (
                "I'm sorry you're experiencing an issue. "
                "This needs help from a human support representative. "
                "Please contact the business support team directly."
            ),
            "escalate": True,
            "reason": "Human review requested",
        }

    matched_products = search_products(question)
    matched_faqs = search_faqs(question)

    # Product prices and stock always come from the CSV.
    if matched_products and (
        is_price_question(question) or is_stock_question(question)
    ):
        product = matched_products[0]

        return {
            "answer": format_product(product),
            "escalate": False,
            "reason": "",
        }

    # Answer an exact or strong FAQ match from trusted business data.
    if matched_faqs:
        best_faq = matched_faqs[0]

        if len(tokens(question) & tokens(best_faq["question"])) >= 2:
            return {
                "answer": best_faq["answer"],
                "escalate": False,
                "reason": "",
            }

    # If nothing relevant was found, do not invent business policies.
    if not matched_products and not matched_faqs:
        try:
            save_escalation(question, "No matching business information")
        except OSError as error:
            print(f"Escalation save error: {error}")

        return {
            "answer": (
                "I couldn't find verified information for that question. "
                "Please contact the business support team for assistance."
            ),
            "escalate": True,
            "reason": "No matching business information",
        }

    # Use Gemini only when relevant business context exists.
    if client is None:
        return {
            "answer": (
                "I found some relevant business information, but AI "
                "responses are currently unavailable. Please try asking "
                "about a product's price or stock, or check the business FAQ."
            ),
            "escalate": False,
            "reason": "",
        }

    product_context = []

    for product in matched_products:
        product_context.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
            "price": money(product["price"]),
            "stock": int(product["stock"]),
            "description": product["description"],
        })

    faq_context = [
        {
            "question": faq["question"],
            "answer": faq["answer"],
        }
        for faq in matched_faqs
    ]

    prompt = f"""
You are BizzBudy, a polite customer-support assistant.

Answer the customer's question using ONLY the business information below.
Do not invent prices, stock, delivery promises, discounts, or policies.
If the supplied information does not answer the question, say so.
Keep the answer clear, friendly, and concise.

BUSINESS PRODUCTS:
{json.dumps(product_context, ensure_ascii=False)}

BUSINESS FAQs:
{json.dumps(faq_context, ensure_ascii=False)}

CUSTOMER QUESTION:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        answer = (response.text or "").strip()

        if not answer:
            raise ValueError("Gemini returned an empty answer.")

        return {
            "answer": answer,
            "escalate": False,
            "reason": "",
        }

    except Exception as error:
        # Keep technical details in server logs, not customer-facing text.
        print(
            f"BizzBudy Gemini error: "
            f"{type(error).__name__}: {error}"
        )

        return {
            "answer": (
                "I'm having trouble generating an AI response right now. "
                "You can still ask about product prices and stock, which "
                "are read directly from our product catalogue."
            ),
            "escalate": False,
            "reason": "Gemini service temporarily unavailable",
        }


# --------------------------------------------------
# 10. USER INTERFACE
# --------------------------------------------------

st.title("🛍️ BizzBudy AI")
st.caption("Your AI-powered business customer-support assistant")

st.markdown(
    "Ask about products, prices, availability, and business FAQs."
)

if data_error:
    st.error(f"Business data error: {data_error}")
    st.info(
        "Place products.csv and faqs.json in the same folder as app.py."
    )

if not API_KEY:
    st.warning(
        "Gemini API key is not configured. Product prices, stock, "
        "and matching FAQs can still work."
    )

# Suggested questions
st.subheader("Try asking")

suggestions = [
    "What is the price of the wireless mouse?",
    "Is the keyboard available?",
    "What are your opening hours?",
    "What payment methods do you accept?",
]

cols = st.columns(2)

for index, suggestion in enumerate(suggestions):
    if cols[index % 2].button(
        suggestion,
        key=f"suggestion_{index}",
        use_container_width=True,
    ):
        st.session_state["pending_question"] = suggestion


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to BizzBudy. "
                "How can I help you today?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("escalate"):
            st.warning("Human support may be needed for this question.")


# Chat input and suggested question handling
pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input("Type your question here...")

question = pending_question or typed_question

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    result = answer_customer(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "escalate": result["escalate"],
    })

    with st.chat_message("assistant"):
        st.markdown(result["answer"])

        if result["escalate"]:
            st.warning(
                "Human support may be needed for this question."
            )

    st.rerun()


# Product catalogue
with st.expander("📦 View product catalogue"):
    if products.empty:
        st.info("No products are available to display.")
    else:
        display_products = products.copy()
        display_products["price"] = display_products["price"].apply(money)
        display_products["stock"] = display_products["stock"].astype(int)
        st.dataframe(
            display_products,
            use_container_width=True,
            hide_index=True,
        )


# FAQ catalogue
with st.expander("❓ Frequently asked questions"):
    if not faqs:
        st.info("No FAQs are available.")
    else:
        for faq in faqs:
            st.markdown(f"**{faq['question']}**")
            st.write(faq["answer"])
            st.divider()


# Escalation log
with st.expander("🧑‍💼 Support escalation records"):
    if ESCALATIONS_FILE.exists():
        try:
            escalation_data = pd.read_csv(ESCALATIONS_FILE)
            st.dataframe(
                escalation_data,
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.info("Escalation records could not be displayed.")
    else:
        st.info("There are no saved escalation records yet.")

st.caption("BizzBudy AI · Verify business policies before using live.")
