import streamlit as st
import gradio as gr
import pandas as pd
from multiprocessing import Process
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Function to set up RAG chain
def setup_rag_chain(api_key, dfs):
    llm = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-4o")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Prepare documents by extracting dataframes from each sheet
    documents = []
    for sheet_name, df in dfs.items():
        # Convert each dataframe to a single text block
        documents.append(df.to_string(index=False))

    # Split documents into chunks suitable for vector embedding
    split_docs = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]
    vectorstore = Chroma.from_texts(split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 18})

    # Define the prompt for RAG process
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the information provided."),
        ("human", "Question: {input}\nContext: {context}")
    ])

    # Combine documents in retrieval chain
    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_documents_chain)
    return qa_chain


# Gradio Interface Function
def gradio_interface(api_key, dfs):
    qa_chain = setup_rag_chain(api_key, dfs)

    def answer_question(question):
        response = qa_chain.invoke({"input": question})
        return response["answer"]

    # Define Gradio interface without async or queue
    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
        outputs=gr.Textbox(),
        live=True
    )

    # Launch Gradio interface
    interface.launch(share=True)


# Run Gradio in a separate process
def start_gradio(api_key, dfs):
    gradio_process = Process(target=gradio_interface, args=(api_key, dfs))
    gradio_process.start()
    return gradio_process


# Streamlit Main Function
def main():
    st.title("Excel Document Q&A System")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if api_key and uploaded_file:
        # Load each sheet in the Excel file as a separate dataframe
        dfs = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl", header=None)

        # Start Gradio in a separate process
        gradio_process = start_gradio(api_key, dfs)

        # Optionally, stop Gradio process when Streamlit session ends
        st.write("Gradio interface launched. Access the link above for Q&A.")
        st.session_state["gradio_process"] = gradio_process


if __name__ == "__main__":
    main()
