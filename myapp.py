import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Prompts
prompt = ChatPromptTemplate.from_template(
    """ Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer . 
    I will tip you 1000$ if the user finds the answer helpful.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

intent_prompt = ChatPromptTemplate.from_template(
    """You are an AI designed to assist with inquiries. A user has asked the following question: "{input}". 
    Is the user asking to be contacted (such as requesting a call, email, or other forms of communication)? 
    Answer with "Yes" or "No" only.
    """
)

info_verification_prompt = ChatPromptTemplate.from_template(
    """A user has provided the following information: "{input}". 
    Does this message contain the user's full name, phone number, and email address? 
    If not, specify which information is missing or incomplete. Otherwise, Respond in the following format:
    Name: [name]
    Phone: [phone]
    Email: [email]
    """
)

# Function to save user information
def save_info(name, phone, email, csv_file="user_info.csv"):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, phone, email])

# Function to detect user's intent
def detect_intent(query):
    response = llm(intent_prompt.format(input=query))
    return response.strip().lower() == "yes"

# Function to verify user information
def verify_info(query):
    response = llm(info_verification_prompt.format(input=query))
    return response

# Function to extract user information from the response
def extract_info(response):
    name = response.split("Name: ")[1].split("Phone: ")[0].strip() if "Name: " in response else None
    phone = response.split("Phone: ")[1].split("Email: ")[0].strip() if "Phone: " in response else None
    email = response.split("Email: ")[1].strip() if "Email: " in response else None
    
    # Handle missing information
    name = None if name == "None" else name
    phone = None if phone == "None" else phone
    email = None if email == "None" else email
    
    return name, phone, email

# Main function to handle user queries
def handle_user_query(query, vectors, retrieval_chain):
    # Check if the user is requesting contact (e.g., "call me")
    if detect_intent(query):
        print("Bot: Please provide your Name, Phone Number, and Email below in the format: Name: John Doe, Phone: 1234567890, Email: john.doe@example.com")
    else:
        # Verify the provided contact information
        verification_result = verify_info(query)
        if "Name:" in verification_result and "Phone:" in verification_result and "Email:" in verification_result:
            name, phone, email = extract_info(verification_result)
            save_info(name, phone, email)
            print("Bot: Thank you! Your contact information has been saved.")
        elif "Name:" in verification_result or "Phone:" in verification_result or "Email:" in verification_result:
            print(f"Bot: Please provide the following missing information: {verification_result}")
        else:
            # Process the query if it's not related to contact information
            response = retrieval_chain.invoke({'input': query})['answer']
            print(f"Bot: {response}")


# Function to create vector embeddings
def vector_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./pdfs")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)

    retriever_vectordb = vectors.as_retriever(search_kwargs={"k": 4})
    keyword_retriever = BM25Retriever.from_documents(final_documents)
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5])

    documents_chains = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(ensemble_retriever, documents_chains)

    return vectors, retrieval_chain

if __name__ == "__main__":
    vectors, retrieval_chain = vector_embeddings()
    print("FAISS is ready")

    # Initialize conversation state
    collecting_info = False
    info_saved = False

    # Simulated chat loop
    while True:
        user_input = input("Your message: ")
        if user_input.lower() == "exit":
            break
        handle_user_query(user_input, vectors, retrieval_chain)
