from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY  # Your API key from .env or config file
from langchain.chains import RetrievalQA

def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Updated to current supported model
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return rag_chain
