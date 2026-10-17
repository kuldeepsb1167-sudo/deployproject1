import streamlit as st
from google import genai

# ---------------------------------
# 1. Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Pizza Delivery Bot",
    page_icon="🍕"
)

st.title("🍕 Pizza Delivery Chatbot")
st.write("Welcome! Order your favorite pizza or ask us a question.")

# ---------------------------------
# 2. Predefined menu dictionary
# ---------------------------------
pizza_menu = {
    "Margherita": 199,
    "Farmhouse": 299,
    "Veggie Supreme": 349,
    "Paneer Tikka": 329,
    "Chicken Delight": 379
}

# ---------------------------------
# 3. Predefined questions and answers
# ---------------------------------
faq = {
    "hello": "Hello! Welcome to Pizza Hub. How can I help you?",
    "hi": "Hi! Welcome to Pizza Hub. Would you like to see our menu?",
    "menu": "Our pizzas: Margherita Rs.199, Farmhouse Rs.299, Veggie Supreme Rs.349, Paneer Tikka Rs.329, Chicken Delight Rs.379.",
    "delivery": "Our standard delivery estimate is 30-45 minutes. Actual delivery time may vary.",
    "payment": "You can choose cash on delivery or an available online payment option at checkout.",
    "offers": "Ask our staff about current offers. We do not have a verified live offers list in this demo.",
    "address": "Please provide your delivery address when placing your order.",
    "order status": "This demo does not connect to live orders. Please contact the store to check your order status.",
    "hours": "Please contact Pizza Hub to confirm today's opening hours.",
    "contact": "Please contact your local Pizza Hub store for assistance.",
    "thank you": "You're welcome! Enjoy your pizza!",
    "bye": "Thank you for visiting Pizza Hub. Have a delicious day!"
}

# ---------------------------------
# 4. API configuration
# ---------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Choose model",
        ["gemini-3.1-flash-lite"]
    )

    st.subheader("🍕 Pizza Menu")

    for pizza, price in pizza_menu.items():
        st.write(f"{pizza} — Rs. {price}")

# ---------------------------------
# 5. Display predefined menu
# ---------------------------------
with st.expander("View Pizza Menu"):
    for pizza, price in pizza_menu.items():
        st.write(f"🍕 **{pizza}** — Rs. {price}")

# ---------------------------------
# 6. Initialize chat history
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------
# 7. Find predefined answers
# ---------------------------------
def find_predefined_answer(question):
    q = question.lower().strip()

    # Match complete predefined question keywords
    for keyword, answer in faq.items():
        if keyword in q:
            return answer

    return None

# ---------------------------------
# 8. Chat input and LLM response
# ---------------------------------
question = st.chat_input("Ask about pizza, menu, delivery...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Check predefined answers first
    answer = find_predefined_answer(question)

    if answer is None:
        if not api_key:
            answer = (
                "Please enter your Gemini API key in the sidebar "
                "to answer questions outside our predefined FAQ."
            )
        else:
            try:
                client = genai.Client(api_key=api_key)

                instructions = f"""
                You are a friendly Pizza Hub delivery assistant.

                Use only these verified menu items and prices:
                {pizza_menu}

                Predefined FAQ information:
                {faq}

                Rules:
                - Help customers understand the menu and choose pizzas.
                - Be friendly, concise, and helpful.
                - Never invent prices, discounts, delivery availability,
                  live order status, or store policies.
                - If information is unavailable, say the customer
                  should contact the store.
                - Do not claim an order has actually been placed.
                - Ask the customer which pizza and quantity they want
                  if they are interested in ordering.
                """

                # Include recent chat messages for context
                history = st.session_state.messages[-10:]

                conversation = [
                    {"role": "system", "content": instructions}
                ]

                for msg in history:
                    conversation.append({
                        "role": (
                            "user" if msg["role"] == "user"
                            else "assistant"
                        ),
                        "content": msg["content"]
                    })

                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        f"{msg['role'].upper()}: {msg['content']}"
                        for msg in conversation
                    ]
                )

                answer = response.text or (
                    "Sorry, I could not generate a response."
                )

            except Exception as e:
                answer = (
                    "Unable to connect to Gemini. Check your API key, "
                    "model availability, internet, and API quota."
                )
                st.error(f"Error details: {e}")

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ---------------------------------
# 9. Clear conversation
# ---------------------------------
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()