

# Chatbot for Nepal Demographic and Health Survey 2022

## Overview

This project implements a chatbot that leverages Google Generative AI and embeddings to interact with users and provide insights based on the **Nepal Demographic and Health Survey 2022**. The chatbot can process user queries, collect contact information, and save it to a CSV file for further inquiries. The system utilizes a hybrid retrieval-augmented generation (RAG) search technique to deliver accurate and contextually relevant answers.

## Features

- **Google Generative AI Integration**: The chatbot uses Google Generative AI for generating responses to user queries. It leverages the `GoogleGenerativeAI` model for natural language understanding and response generation.
  
- **Google Generative AI Embeddings**: For embedding textual data, the system uses `GoogleGenerativeAIEmbeddings`, which allows for the creation of high-quality vector embeddings from the survey data.

- **Hybrid RAG Search Technique**: The chatbot employs a hybrid retrieval-augmented generation (RAG) search approach that combines vector-based retrieval using FAISS with traditional keyword-based retrieval using BM25. This hybrid method ensures both semantic relevance and keyword accuracy in the responses provided by the chatbot.

- **User Information Collection**: The chatbot can detect when a user is providing contact information (such as name, phone number, and email) and save this information to a CSV file (`user_info.csv`) for further follow-up or inquiries.

## What is Hybrid RAG Search?

Hybrid Retrieval-Augmented Generation (RAG) is a sophisticated technique used in modern AI systems to enhance the accuracy and relevance of responses. It involves two main components:

1. **Retrieval Component**: 
   - **Vector-Based Retrieval**: This component uses vector embeddings (in this case, from Google Generative AI Embeddings) to perform a semantic search, retrieving documents that are semantically similar to the user's query.
   - **Keyword-Based Retrieval**: In parallel, a traditional keyword-based retrieval method (BM25) is used to ensure that the most relevant documents containing specific keywords from the query are also considered.

2. **Augmented Generation**: 
   - The results from both retrieval methods are combined and fed into the generative AI model to produce a coherent, contextually relevant response. This ensures that the chatbot can generate responses that are both accurate in terms of content and relevant to the user's query.

The hybrid approach ensures a robust search capability, effectively combining the strengths of semantic understanding and traditional keyword matching.

## Installation

To set up and run the chatbot locally, follow these steps:

1. **Clone the Repository**:
   ```python
   git clone https://github.com/Puzan789/Nepal_demographic_health_survey_chatbot.git
   
   ```

2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Run the Application**:
   Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```

## Usage

- **Interacting with the Chatbot**: 
  - Open the chatbot interface in your browser (it will launch automatically when you run the application).
  - Type your query related to the **Demographic and Health Survey 2022** into the input field.
  - The chatbot will respond based on the retrieved and generated content.

- **Providing Contact Information**:
  - If the chatbot asks for your contact information, you can provide it in the format: `Name: John Doe, Phone: 1234567890, Email: john.doe@example.com`.
  - The information will be saved to `user_info.csv` for future reference.

## Conclusion

This project demonstrates the power of hybrid RAG search combined with Google Generative AI and embeddings in creating a sophisticated, responsive chatbot interface. The chatbot provides a reliable way to interact with the **Demographic and Health Survey 2022** data.

