import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
import re
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Check if required environment variables are set
if not os.getenv('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY environment variable not set")

def scrape_website_content(url):
    """
    Scrapes the website content from the given URL.
    """
    print(f"Attempting to scrape content from {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = "\n".join([para.get_text() for para in paragraphs])
            print(f"Successfully scraped content from {url}")
            return content
        else:
            print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error occurred while scraping {url}: {e}")
        return ""

def process_website(url):
    """
    Processes the website: scrapes text, splits it into chunks,
    and converts chunks into embeddings.
    """
    print(f"Processing website: {url}")
    text = scrape_website_content(url)

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks")

    chunks_with_sources = [(chunk, {"source": url}) for chunk in chunks]
    return chunks_with_sources

def upload_website_data(url):
    """
    Scrapes the website, processes the content, and saves it into a FAISS vector store.
    """
    print(f"Uploading website data for {url}")
    chunks_with_sources = process_website(url)
    if chunks_with_sources:
        text_chunks, metadata = zip(*chunks_with_sources)
        print(f"Creating embeddings for {len(text_chunks)} chunks")
        embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv('GOOGLE_API_KEY'), model="models/text-embedding-004")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
        vector_store.save_local("faiss_index")
        print("FAISS index created or updated successfully.")
    else:
        print("No valid website data to process.")

def reframe_with_gemini(text,question):
    # Configure the API key from the environment variable
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Set up the generation configuration
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 1200,

    }
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    # Prepare the prompt
    prompt = f"""
You are a chatbot designed to answer questions using website content. Answer the question directly using the provided website text.

User Query: {question}
Website Text: {text}

Ensure that the response:
1. Addresses the user's question directly.
2. Politely suggests more specific queries if the answer isn't found.
3. Encourages the user to consult more information if needed.
4. Avoids guessing irrelevant information.
5.  Do not suggest exploring the website manually or can also find more information on their website., as the main objective is to provide information directly from the available content.
"""

    # Generate the response
    try:
        response = model.generate_content(prompt)
        
        # Extract and return the content
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            print("No candidates found in the response.")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def generate_natural_language_response(relevant_info, question):
    """
    Generates a natural language response based on the relevant information.
    """
    if not relevant_info:
        return "Sorry, I couldn't find any relevant information."

    response = "Here's what I found based on your question:\n\n"
    
    # Aggregate and format the information
    aggregated_texts = []
    for text, _ in relevant_info:
        aggregated_texts.append(text)

    combined_text = " ".join(aggregated_texts)  # Combine all relevant texts
    summarized_text = reframe_with_gemini(combined_text, question)  # Reframe combined text
    response += summarized_text

    return response.strip()


def extract_relevant_information(question, text_chunks, metadata):
    """
    Extracts and aggregates relevant information based on the question.
    """
    relevant_info = []
    keywords = re.findall(r'\b\w+\b', question.lower())
    
    for chunk, meta in zip(text_chunks, metadata):
        chunk_lower = chunk.lower()
        if any(keyword in chunk_lower for keyword in keywords):
            relevant_info.append((chunk, meta))
    
    return relevant_info

def query(question, chat_history):
    """
    Processes a query using the conversational retrieval chain and returns a natural language response.
    """
    try:
        # Initialize embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv('GOOGLE_API_KEY'), model="models/text-embedding-004")
        vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

        # Retrieve the relevant chunks based on the question
        search_results = vector_store.similarity_search(question)
        if not search_results:
            return {"answer": "I couldn't find any relevant information.", "sources": []}

        # Extract text and metadata from search results
        text_chunks = [result.page_content for result in search_results]
        metadata = [result.metadata for result in search_results]

        # Extract relevant information
        relevant_info = extract_relevant_information(question, text_chunks, metadata)

        # Generate a response using the reframed information
        formatted_answer = generate_natural_language_response(relevant_info,question) if relevant_info else "I couldn't find a specific answer. Could you please provide more details or ask a different question?"

        return {"answer": formatted_answer}
    
    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"answer": "Oops, something went wrong while processing your query. Please try again later."}



def show_ui():
    """
    Sets up the Streamlit UI for the Analytx4T Personal Chatbot.
    """
    st.title("Analytx4T Personal Chatbot")
    st.image("https://analytx4t.com/wp-content/uploads/2020/03/final.png", width=300)  # Analytx4T logo
    st.write("Hello! I am your Analytx4T Personal Chatbot. How can I assist you today?")

    # Initialize session state for chat history and messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter Your Query"):
        try:
            # Process the user's query and get the response
            with st.spinner("Processing your query..."):
                response = query(question=prompt, chat_history=st.session_state.chat_history)

                # Display the user's query and bot's response
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(f"{response.get('answer', 'Sorry, I couldn\'t find an answer.')}")

                # Store the messages in the session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response.get('answer', 'Sorry, I couldn\'t find an answer.')})

                # Update chat history
                st.session_state.chat_history.append((prompt, response.get('answer', 'Sorry, I couldn\'t find an answer.')))
        except Exception as e:
            st.error(f"An error occurred while processing the query: {str(e)}")

    # Add a "Restart Chat" button below the output area and above the input area
    if st.button("Restart Chat"):
        # Reset the chat state
        st.session_state.messages = []
        st.session_state.chat_history = []



if __name__ == "__main__":
    website_url = "https://analytx4t.com/"  # Change to your desired website URL
    print("Starting to upload website data")
    upload_website_data(website_url)  # Scrape and process the website
    print("Launching Streamlit UI")
    show_ui()