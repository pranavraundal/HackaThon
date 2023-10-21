import streamlit as st 
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader as PdfRead
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv 
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback 

with st.sidebar:
    st.title("PDF SE BAAT-CHIT")
    add_vertical_space(5)

def main():
    st.header("Chat with PDF")
    pdf = st.file_uploader("Upload PDF", type =["pdf"])
    if pdf is not None:
        pdf_reader = PdfRead(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_text(text=text) 
        # embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4] 
        if os.path.exists(f"{store_name}.pkl"):
            with open (f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f) 
        else:        
            embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY,temperature=1)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open (f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        que = st.text_input("Enter your question")
        if que:
            doc = VectorStore.similarity_search(que,k=3 )
            llm = OpenAI(openai_api_key=OPEN_API_KEY,model_name="gpt-3.5-turbo",temperature=1)
            chain = load_qa_chain(llm,chain_type="stuff")
            with get_openai_callback() as cb :
                response = chain.run(input_documents=doc, question=que)
                print(cb )
            st.write(response)
            # st.write(doc)
        # st.write(chunks)
# if __name__ == "__main__":
    
#     load_dotenv()
#     OPEN_API_KEY = os.getenv("OPEN_API_KEY")
#     main()  
load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
main()