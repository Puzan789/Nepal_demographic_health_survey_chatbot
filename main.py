import streamlit as st
from streamlit_chat import message as st_message
import os
from dotenv import load_dotenv
from  myapp import vector_embeddings, handle_user_query  

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Streamlit page configuration
st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")

if "vectors" not in st.session_state or "retrieval_chain" not in st.session_state:
    st.write("Initializing vectors and retrieval chain... This may take a moment.")
    vectors, retrieval_chain = vector_embeddings()
    st.session_state.vectors = vectors
    st.session_state.retrieval_chain = retrieval_chain
    st.write("Initialization complete!")

if "history" not in st.session_state:
    st.session_state.history = []

# Function to handle chat
def chat_with_bot(user_input):
    # Add user message to the session state
    st.session_state.history.append({"message": user_input, "is_user": True})
    response = handle_user_query(user_input, st.session_state.vectors, st.session_state.retrieval_chain)
    st.session_state.history.append({"message": response, "is_user": False})

# Display the conversation
for i, chat in enumerate(st.session_state.history):
    st_message(chat["message"], is_user=chat["is_user"], key=str(i))

# Input field for user to type their message
user_input = st.text_input("Your message: ", "")

# Handle the user's message when the form is submitted
if user_input:
    chat_with_bot(user_input)
    # Clear the input field after submission
    st.experimental_rerun()

# Option to reset conversation
if st.button("Reset Conversation"):
    st.session_state.history = []
    st.experimental_rerun()
