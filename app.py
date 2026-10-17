
import os
import re
import json
from datetime import datetime
from pathlib import Path

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
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_FILE = BASE_DIR / "products.csv"
FAQS_FILE = BASE_DIR / "faqs.json"
DATA_DIR = BASE_DIR / "data"
ESCALATION_FILE = DATA_DIR / "escalations.csv"

MODEL_NAME = "gemini-3.8-flash"

load_dotenv(BASE_DIR / ".env")


def get_api_key():
    """Read the API key from Streamlit Secrets or a local .env file."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


API_KEY = get_api_key()


# --------------------------------------------------
# 2. LOAD PRODUCT CATALOGUE
# --------------------------------------------------

@st.cache_data
def load_products():
    if not PRODUCTS_FILE.exists():
        return pd.DataFrame(
            columns=[
                "product_id",
                "name",
                "category",
                "price",
                "stock",
                "description",
            ]
        )

    df = pd.read_csv(PRODUCTS_FILE)

    required_columns = [
        "product_id",
        "name",
        "category",
        "price",
        "stock",
        "description",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    df["name"] = df["name"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0)

    return df


products = load_products()


# --------------------------------------------------
# 3. LOAD FAQs
# --------------------------------------------------

@st.cache_data
def load_faqs():
    if not FAQS_FILE.exists():
        return []

    try:
        with open(FAQS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if isinstance(data.get("faqs"), list):
                return data["faqs"]

            return [
                {"question": question, "answer": answer}
                for question, answer in data.items()
                if isinstance(answer, str)
            ]

        return []

    except (json.JSONDecodeError, OSError):
        return []


faqs = load_faqs()


# --------------------------------------------------
# 4. INITIALIZE GEMINI
# --------------------------------------------------

@st.cache_resource
def get_gemini_client(api_key):
    if not api_key:
        return None

    return genai.Client(api_key=api_key)


client = get_gemini_client(API_KEY) if API_KEY else None


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

def format_money(amount):
    """Format prices in Indian rupees."""
    return f"₹{float(amount):,.0f}"


def normalize_text(text):
    """Normalize text for simple keyword matching."""
    return re.sub(r"[^a-z0-9₹\s]", " ", str(text).lower()).strip()


def search_products(question):
    """Find catalogue products related to the question."""
    if products.empty:
        return products.copy()

    question_words = set(normalize_text(question).split())

    stop_words = {
        "what", "is", "the", "a", "an", "of", "for", "do",
        "you", "have", "show", "me", "please", "available",
        "price", "cost", "tell", "about", "can", "i", "buy",
        "in", "stock", "your", "product", "products",
    }

    question_words -= stop_words

    if not question_words:
        return products.iloc[0:0].copy()

    matches = []

    for _, product in products.iterrows():
        searchable_text = normalize_text(
            f"{product['name']} {product['category']} "
            f"{product['description']} {product['product_id']}"
        )

        product_words = set(searchable_text.split())
        score = len(question_words & product_words)

        if score > 0:
            matches.append((score, product))

    matches.sort(key=lambda item: item[0], reverse=True)

    if not matches:
        return products.iloc[0:0].copy()

    return pd.DataFrame(
        [item[1] for item in matches]
    ).reset_index(drop=True)


def search_products_under_budget(question):
    """Find products within a budget mentioned by the customer."""
    text = question.lower().replace(",", "")

    patterns = [
        r"(?:under|below|less than|within|up to|upto)\s*₹?\s*(\d+(?:\.\d+)?)",
        r"₹\s*(\d+(?:\.\d+)?)\s*(?:or less|and below|max(?:imum)?)",
        r"(?:budget|price limit)\s*(?:of|is|:)?\s*₹?\s*(\d+(?:\.\d+)?)",
    ]

    budget = None

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            budget = float(match.group(1))
            break

    if budget is None or products.empty:
        return None

    results = products[products["price"] <= budget].copy()
    results = results.sort_values("price")

    if results.empty:
        return f"No products were found at or below {format_money(budget)}."

    lines = [
        f"### Products available under {format_money(budget)}",
        "",
    ]

    for _, product in results.iterrows():
        stock = int(product["stock"])
        availability = (
            f"{stock} in stock" if stock > 0 else "Out of stock"
        )

        lines.append(
            f"- **{product['name']}** — "
            f"{format_money(product['price'])} "
            f"({availability})"
        )

    return "\n".join(lines)


def search_faqs(question):
    """Return the best matching FAQ, if one is found."""
    question_words = set(normalize_text(question).split())

    stop_words = {
        "what", "is", "the", "a", "an", "do", "does", "how",
        "can", "i", "you", "your", "me", "my", "about", "please",
    }

    question_words -= stop_words

    best_faq = None
    best_score = 0

    for faq in faqs:
        faq_question = str(faq.get("question", ""))
        faq_answer = str(faq.get("answer", ""))

        faq_words = set(normalize_text(faq_question).split())
        score = len(question_words & faq_words)

        if score > best_score:
            best_score = score
            best_faq = {
                "question": faq_question,
                "answer": faq_answer,
                "score": score,
            }

    # Require at least one meaningful keyword match.
    if best_faq and best_score >= 1:
        return best_faq

    return None


def requires_human_support(question):
    """Detect requests that should be escalated to a human."""
    keywords = [
        "speak to human",
        "talk to human",
        "human agent",
        "real person",
        "customer service agent",
        "manager",
        "complaint",
        "fraud",
        "scam",
        "chargeback",
        "legal action",
        "damaged product",
        "wrong product",
        "broken product",
        "refund not received",
        "payment deducted",
        "unauthorized payment",
    ]

    text = question.lower()

    return any(keyword in text for keyword in keywords)


def save_escalation(question):
    """Save a support request without storing API keys or secrets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "customer_question": question,
                "status": "Pending",
            }
        ]
    )

    record.to_csv(
        ESCALATION_FILE,
        mode="a",
        header=not ESCALATION_FILE.exists(),
        index=False,
        encoding="utf-8",
    )


# --------------------------------------------------
# 6. GEMINI RESPONSE
# --------------------------------------------------

def ask_gemini(question, history=None, product_context=None, faq_context=None):
    """Ask Gemini to answer using available business information."""
    if client is None:
        return (
            "The AI service is not configured yet. "
            "Please add a valid GEMINI_API_KEY to your local .env "
            "file or your Streamlit Cloud Secrets."
        )

    context_parts = []

    if product_context:
        context_parts.append(
            "Relevant catalogue information:\n" + product_context
        )

    if faq_context:
        context_parts.append(
            "Relevant FAQ information:\n" + faq_context
        )

    context = "\n\n".join(context_parts) or "No matching catalogue or FAQ information was found."

    prompt = f"""
You are BizzBudy AI, a polite and helpful shopping assistant.

Rules:
- Answer clearly and concisely.
- Use the supplied catalogue and FAQ information when relevant.
- Never invent product prices, stock counts, delivery promises, warranty terms,
  return policies, discounts, or business policies.
- If the supplied information does not answer a policy question, say that
  the customer should confirm the details with customer support.
- You may answer general questions, but stay helpful and professional.
- Do not claim that an order has been placed, cancelled, refunded, or tracked
  unless that information is actually provided.

Business information:
{context}

Customer question:
{question}
"""

    if history:
        recent_history = history[-6:]
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in recent_history
        )
        prompt += f"\n\nRecent conversation:\n{history_text}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if answer and answer.strip():
            return answer.strip()

        return (
            "I couldn't generate a response for that question. "
            "Please try rephrasing it or contact customer support."
        )

    except Exception as error:
        # Do not print or display API keys.
        print(
            f"BizzBudy Gemini error: "
            f"{type(error).__name__}: {error}"
        )

        return (
            "Sorry, the AI service is temporarily unavailable. "
            "I can still check product prices, stock, budget ranges, "
            "and matching FAQs. Please try your question again shortly."
        )


# --------------------------------------------------
# 7. CUSTOMER QUESTION HANDLER
# --------------------------------------------------

def answer_customer(question, history=None):
    """Choose a direct catalogue/FAQ answer or ask Gemini."""
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "escalate": False,
        }

    # Human escalation takes priority.
    if requires_human_support(question):
        try:
            save_escalation(question)
            answer = (
                "I've recorded your request for human support. "
                "A support team member will need to review it. "
                "This demo does not automatically send a notification "
                "to a real support team."
            )
        except OSError:
            answer = (
                "This request needs human support, but I couldn't save "
                "the escalation record. Please contact the business "
                "through its official support channel."
            )

        return {"answer": answer, "escalate": True}

    # Answer budget questions directly from the CSV.
    budget_answer = search_products_under_budget(question)

    if budget_answer is not None:
        return {
            "answer": budget_answer,
            "escalate": False,
        }

    # Find products related to the question.
    matched_products = search_products(question)

    # Direct product-price/stock answer.
    product_question = any(
        term in question.lower()
        for term in [
            "price", "cost", "how much", "stock",
            "available", "availability",
        ]
    )

    if product_question and not matched_products.empty:
        lines = ["### Matching products", ""]

        for _, product in matched_products.head(5).iterrows():
            stock = int(product["stock"])
            availability = (
                f"{stock} in stock" if stock > 0 else "Out of stock"
            )

            lines.append(
                f"- **{product['name']}** — "
                f"{format_money(product['price'])}; "
                f"{availability}. "
                f"{product['description']}"
            )

        return {
            "answer": "\n".join(lines),
            "escalate": False,
        }

    # Search FAQ information.
    faq_match = search_faqs(question)

    if faq_match and faq_match["score"] >= 2:
        return {
            "answer": (
                f"**FAQ: {faq_match['question']}**\n\n"
                f"{faq_match['answer']}"
            ),
            "escalate": False,
        }

    # Prepare supporting context for Gemini.
    product_context = None

    if not matched_products.empty:
        product_lines = []

        for _, product in matched_products.head(5).iterrows():
            product_lines.append(
                f"{product['name']} | "
                f"Category: {product['category']} | "
                f"Price: {format_money(product['price'])} | "
                f"Stock: {int(product['stock'])} | "
                f"Description: {product['description']}"
            )

        product_context = "\n".join(product_lines)

    faq_context = None

    if faq_match:
        faq_context = (
            f"Question: {faq_match['question']}\n"
            f"Answer: {faq_match['answer']}"
        )

    answer = ask_gemini(
        question=question,
        history=history,
        product_context=product_context,
        faq_context=faq_context,
    )

    return {
        "answer": answer,
        "escalate": False,
    }


# --------------------------------------------------
# 8. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🛍️ BizzBudy AI")
st.caption("Your shopping assistant for products, prices, stock, and FAQs.")

with st.sidebar:
    st.header("About BizzBudy")
    st.write(
        "Ask questions about products, prices, stock, budgets, "
        "and the business FAQs."
    )

    st.subheader("AI service status")

    if client is not None:
        st.success("API key configured")
        st.caption(
            "This confirms that a key was found, not that Gemini "
            "requests are succeeding."
        )
    else:
        st.warning("API key not configured")

    st.subheader("Catalogue summary")

    if products.empty:
        st.warning("No products loaded. Check products.csv.")
    else:
        st.metric("Products", len(products))
        st.metric(
            "Products in stock",
            int((products["stock"] > 0).sum()),
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Suggested questions.
st.subheader("Try asking")

suggestions = [
    "What is the price of the wireless mouse?",
    "What is available under ₹1000?",
    "Which audio products do you have?",
    "What is your warranty policy?",
]

columns = st.columns(2)

for index, suggestion in enumerate(suggestions):
    with columns[index % 2]:
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_question = suggestion


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept typed input or a suggested question.
typed_question = st.chat_input("Ask BizzBudy something...")

pending_question = st.session_state.pop("pending_question", None)
question = typed_question or pending_question


if question:
    # Display and store the user's message.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Provide prior conversation as context.
    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("BizzBudy is thinking..."):
            result = answer_customer(question, history=history)

        st.markdown(result["answer"])

        if result.get("escalate"):
            st.info("Human support escalation requested.")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )


# --------------------------------------------------
# 9. PRODUCT CATALOGUE
# --------------------------------------------------

with st.expander("🛒 Browse product catalogue"):
    if products.empty:
        st.warning("No products found. Make sure products.csv is in the app folder.")
    else:
        display_products = products.copy()
        display_products["price"] = display_products["price"].apply(format_money)
        display_products["stock"] = display_products["stock"].astype(int)

        st.dataframe(
            display_products,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# 10. FAQ LIST
# --------------------------------------------------

with st.expander("❓ Frequently asked questions"):
    if not faqs:
        st.info("No FAQs loaded. Check faqs.json.")
    else:
        for faq in faqs:
            with st.expander(str(faq.get("question", "Question"))):
                st.write(faq.get("answer", "No answer available."))


# --------------------------------------------------
# 11. ESCALATION STATUS
# --------------------------------------------------

with st.expander("📩 Support escalation status"):
    if ESCALATION_FILE.exists():
        try:
            escalation_df = pd.read_csv(ESCALATION_FILE)
            st.dataframe(
                escalation_df,
                use_container_width=True,
                hide_index=True,
            )
        except (OSError, pd.errors.ParserError):
            st.warning("Unable to read escalation records.")
    else:
        st.caption("No support requests have been recorded yet.")


st.caption("BizzBudy AI • Demo application")
