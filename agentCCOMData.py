import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from io import BytesIO
from fpdf import FPDF
import docx
import logging

# Adding ChromaDB Connection
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

logging.basicConfig(level=logging.INFO)

# Configuration for ChromaDB
configuration = {
    "client": "PersistentClient",
    "path": "chromaData"  # Adjust the path to your server/environment setup
}

collection_name = "my_collection"

# Function to ensure dataframe compatibility with Arrow
def ensure_arrow_compatibility(df):
    """Ensure dataframe is compatible with Arrow by converting all columns to string."""
    return df.astype(str)


# Function to rename duplicate columns without inplace assignment
def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    cols = cols.fillna('Unnamed')
    cols = cols.apply(lambda x: x if 'Unnamed' not in x else None)  # Remove "Unnamed" columns
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        cols.iloc[dup_indices] = [f"{dup}_{i}" for i in range(1, len(dup_indices) + 1)]
    df.columns = cols
    return df.dropna(axis=1, how='all')  # Drop empty or unnamed columns


# Function to extract the client name from the dataframe
def extract_client_name(dfs):
    """Attempt to extract the client name from the loaded sheets."""
    for df in dfs:
        for col in df.columns:
            if 'Client' in col:
                client_name = df[col].iloc[0]  # Usually, the client name is the first row.
                return client_name
    return None  # Fallback if client name not found


# Function to prompt user for OpenAI API key and store it in session state
def get_openai_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    # Prompt user to input the OpenAI API key
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        st.session_state["api_key"] = api_key
        st.success("API Key set successfully!")
    else:
        st.warning("Please enter your OpenAI API key.")

    return st.session_state["api_key"]


@st.cache_data
def load_excel_file(uploaded_file):
    try:
        dfs = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl", header=None)
        processed_dfs = []
        total_sheets = len(dfs)
        sheets_processed = 0

        for sheet_name, df in dfs.items():
            if len(df) < 16:
                st.sidebar.warning(f"Skipping sheet {sheet_name} because it has fewer than 16 rows.")
                continue

            try:
                header_found = False
                for i in range(20):
                    if set(df.iloc[i]).intersection(
                            {'LN#', 'PROGRAM', 'DATE', 'COST', 'EST (000)', 'ACT (000)', 'IDX'}):
                        df.columns = df.iloc[i]
                        df = df.drop(index=list(range(i + 1)))
                        header_found = True
                        break

                if header_found:
                    df = rename_duplicate_columns(df)
                    df = df.fillna('')
                    processed_dfs.append(df)
                    sheets_processed += 1
                    st.sidebar.write(f"Processed sheet {sheet_name} ({sheets_processed}/{total_sheets})")
                else:
                    st.sidebar.warning(f"Processing unstructured data from sheet {sheet_name}.")

            except Exception as sheet_error:
                st.sidebar.error(f"Error processing sheet {sheet_name}: {sheet_error}")
                continue

        if not processed_dfs:
            st.sidebar.error("No usable data found in the uploaded file.")
            return None

        return processed_dfs

    except Exception as e:
        st.sidebar.error(f"Error processing Excel file: {e}")
        return None


def setup_rag_chain(api_key, dfs):
    llm = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-4o")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    documents = []
    for df in dfs:
        doc_text = df.to_string(index=False)  # Converts the entire dataframe into a single text document
        documents.append(doc_text)

    split_docs = []
    for doc in documents:
        split_docs.extend(text_splitter.split_text(doc))

    vectorstore = Chroma.from_texts(split_docs, embedding=embeddings)
    logging.info(f"Total documents for retrieval: {len(split_docs)}")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 18})

    # Use the revised prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert in Nielsen score TV ratings data analysis. Only use the information from the retrieved documents (context) to answer the user's question. "
            "If the answer is not explicitly in the document, state: 'The information is not available in the provided documents.'"
        ),
        HumanMessagePromptTemplate.from_template("Question: {input}\nContext: {context}")
    ])

    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_documents_chain
    )
    return qa_chain


# Answer simple queries directly from the dataframe
def answer_simple_question(prompt, dfs):
    if "client name" in prompt.lower():
        client_name = extract_client_name(dfs)
        if client_name:
            return f"The client name is {client_name}."
        return "Client name not found in the data."

    if "how many tv stations" in prompt.lower():
        for df in dfs:
            if 'STATION' in df.columns:
                num_stations = df['STATION'].nunique()
                return f"There are {num_stations} unique TV stations in the dataset."
    return None


def format_response_data(result):
    """Format the extracted data for better display in the chat."""
    if isinstance(result, dict):
        if 'answer' in result:
            return f"**Answer:** {result['answer']}"
        elif 'output' in result:
            return f"**Answer:** {result['output']}"
        else:
            return "Sorry, I couldn't process that information."
    elif isinstance(result, str):
        return result.strip()
    elif isinstance(result, list):
        formatted_data = "\n".join([f"- {item}" for item in result])
        return formatted_data
    else:
        return str(result)


# Export chat history as PDF
def export_pdf(chat_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for message in chat_history:
        pdf.multi_cell(200, 10, f"{message['role'].capitalize()}: {message['content']}")

    pdf_output = BytesIO()
    pdf.output(pdf_output, "F")
    pdf_output.seek(0)  # Reset the buffer's position to the start
    return pdf_output


# Export chat history as Word
def export_word(chat_history):
    doc = docx.Document()
    for message in chat_history:
        doc.add_paragraph(f"{message['role'].capitalize()}: {message['content']}")

    doc_output = BytesIO()
    doc.save(doc_output)
    doc_output.seek(0)
    return doc_output


# Main function for Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="CCOM Data")
    st.title("CCOM Data")

    openai_api_key = get_openai_api_key()
    if not openai_api_key:
        st.write("Please enter an OpenAI API Key to proceed.")
        return

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # ChromaDB connection for retrieving documents collection
    conn = st.connection("chromadb",
                         type=ChromadbConnection,
                         **configuration)
    documents_collection_df = conn.get_collection_data(collection_name)
    st.dataframe(documents_collection_df)

    dfs = None

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file:
            with st.spinner("Processing your file..."):
                dfs = load_excel_file(uploaded_file)
                if dfs:
                    st.success("File processed.")

    if dfs:
        # Ensure only one accordion
        with st.expander("Structured Data from Processed Sheets", expanded=False):
            for i, df in enumerate(dfs):
                df_arrow_compatible = ensure_arrow_compatibility(df)
                st.dataframe(df_arrow_compatible, height=200)
        qa_chain = setup_rag_chain(openai_api_key, dfs)
        if qa_chain:
            if prompt := st.chat_input("Type your question here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner('Generating response...'):
                        # Initialize formatted_response
                        formatted_response = "Sorry, I couldn't process that information."

                        try:
                            # Answer simple questions first
                            simple_response = answer_simple_question(prompt, dfs)
                            if simple_response:
                                message_placeholder.markdown(f"**Answer:** {simple_response}")
                            else:
                                # Use the RAG pipeline for complex answers
                                response = qa_chain.invoke({"input": prompt})
                                logging.info(f"Raw model response: {response}")

                                # Extract the answer portion of the response
                                formatted_response = format_response_data(response)
                                message_placeholder.markdown(formatted_response)

                        except Exception as e:
                            response = f"An error occurred: {str(e)}"
                            message_placeholder.markdown(response)
                            logging.error(f"Error occurred during response generation: {e}")

                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})

        # Export chat history buttons
        if st.session_state.messages:
            if st.button("Export chat history as Word"):
                doc_output = export_word(st.session_state['messages'])
                st.download_button("Download Word", doc_output, "chat_history.docx")


if __name__ == "__main__":
    main()
