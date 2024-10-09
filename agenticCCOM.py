import streamlit as st
import pandas as pd
import platform
import keyring
from keyring.backends import SecretService
from keyring.backends import macOS  # Correct macOS backend
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import matplotlib.pyplot as plt
import logging
from io import BytesIO
import docx
from fpdf import FPDF

logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # Initialize messages as an empty list


# Function to detect platform and configure keyring accordingly
def configure_keyring():
    if platform.system() == "Darwin":  # macOS
        keyring.set_keyring(macOS.Keyring())  # Correct MacOS backend
    elif platform.system() == "Linux":  # Ubuntu
        keyring.set_keyring(SecretService.Keyring())  # Correct Ubuntu backend
    else:
        st.error("Unsupported platform for keyring authentication.")


# Function to retrieve OpenAI API key from macOS or Ubuntu's keyring
def get_openai_api_key():
    configure_keyring()  # Configure keyring based on the platform
    try:
        api_key = keyring.get_password("openai", "api_key")
        if not api_key:
            st.warning("No API key found. Please set your OpenAI API key.")
        return api_key
    except Exception as e:
        st.error(f"Failed to retrieve API key: {e}")
        return None


# RAG chain setup for LangChain v0.3 and above
def setup_rag_chain(api_key, dfs):
    llm = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-4o")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    documents = []
    for df in dfs:
        doc_text = df.to_string(index=False)  # Convert entire dataframe to a text document
        documents.append(doc_text)

    split_docs = []
    for doc in documents:
        split_docs.extend(text_splitter.split_text(doc))

    vectorstore = Chroma.from_texts(split_docs, embedding=embeddings)
    logging.info(f"Total documents for retrieval: {len(split_docs)}")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 17})

    # Optimized prompt for inference
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert in Nielsen score TV ratings data analysis. Use only the information from the retrieved documents (context) to answer the user's question. "
            "If the answer is explicitly present in the document, provide it directly. If the answer can be logically inferred from the context (e.g., counting items, deducing relationships), use logical reasoning to infer the answer. "
            "If the answer cannot be found or inferred from the document, state: 'The information is not available in the provided documents.'"
        ),
        HumanMessagePromptTemplate.from_template("Question: {input}\nContext: {context}")
    ])

    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_documents_chain
    )
    return qa_chain


# Agent to interact with the dataframe
def create_agent(api_key, df):
    llm = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-4o")
    return create_pandas_dataframe_agent(llm=llm, df=df, allow_dangerous_code=True)


# Answer simple questions directly from the dataframe using an agent
def answer_simple_question(prompt, dfs):
    if "client name" in prompt.lower():
        client_name = extract_client_name(dfs)
        if client_name:
            return f"The client name is {client_name}."
        return "Client name not found in the data."

    if "date range" in prompt.lower():
        for df in dfs:
            if 'DATE' in df.columns or 'Date Range' in df.columns:
                date_range = df[['DATE']].dropna().iloc[0].to_string()
                return f"The date range is {date_range}."

    return None


# Function to extract the client name from the dataframe
def extract_client_name(dfs):
    for df in dfs:
        for col in df.columns:
            if 'Client' in col:
                client_name = df[col].iloc[0]
                return client_name
    return None


# Function to plot data from a dataframe
def plot_data(df):
    if 'STATION' in df.columns and 'COST' in df.columns:
        df_grouped = df.groupby('STATION').sum()
        stations = df_grouped.index
        costs = df_grouped['COST']

        # Plot the data
        fig, ax = plt.subplots()
        ax.bar(stations, costs)
        ax.set_xlabel('TV Stations')
        ax.set_ylabel('Cost')
        ax.set_title('Cost per TV Station')
        st.pyplot(fig)


# Function to format response data for display
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


# Function to export chat history as Word
def export_word(chat_history):
    doc = docx.Document()
    for message in chat_history:
        doc.add_paragraph(f"{message['role'].capitalize()}: {message['content']}")

    doc_output = BytesIO()
    doc.save(doc_output)
    doc_output.seek(0)  # Reset the buffer to the start
    return doc_output


# Function to export chat history as PDF
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


# Main function for Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="CCOM Data")
    st.title("CCOM Data")

    openai_api_key = get_openai_api_key()
    if not openai_api_key:
        st.write("Please enter an OpenAI API Key.")
        return

    dfs = None

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file:
            with st.spinner("Processing your file..."):
                dfs = load_excel_file(uploaded_file)
                if dfs:
                    st.success("File processed.")

    if dfs:
        qa_chain = setup_rag_chain(openai_api_key, dfs)
        if qa_chain:
            agent = create_agent(openai_api_key, dfs[0])  # Create agent using the first DataFrame

            if prompt := st.chat_input("Type your question here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner('Generating response...'):
                        try:
                            # First try to answer simple questions with the agent
                            response = answer_simple_question(prompt, dfs)
                            if response:
                                message_placeholder.markdown(f"**Answer:** {response}")
                            else:
                                # Use RAG for complex queries
                                response = qa_chain.invoke({"input": prompt})
                                message_placeholder.markdown(format_response_data(response))
                        except Exception as e:
                            response = f"An error occurred: {str(e)}"
                            message_placeholder.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

        # Additional functionality to plot data if requested
        if prompt == "plot the data":
            for df in dfs:
                plot_data(df)

    # Export chat history buttons
    if st.session_state['messages']:
        if st.button("Export chat history as PDF"):
            pdf_output = export_pdf(st.session_state['messages'])
            st.download_button(
                label="Download PDF",
                data=pdf_output,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )
        if st.button("Export chat history as Word"):
            word_output = export_word(st.session_state['messages'])
            st.download_button(
                label="Download Word",
                data=word_output,
                file_name="chat_history.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# Function to load the Excel file
@st.cache_data
def load_excel_file(uploaded_file):
    try:
        dfs = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl", header=None)
        processed_dfs = []
        total_sheets = len(dfs)

        for sheet_name, df in dfs.items():
            if len(df) < 16:
                st.sidebar.warning(f"Skipping sheet {sheet_name} because it has fewer than 16 rows.")
                continue

            for i in range(20):
                if set(df.iloc[i]).intersection({'LN#', 'PROGRAM', 'DATE', 'COST', 'EST (000)', 'ACT (000)', 'IDX'}):
                    df.columns = df.iloc[i]
                    df = df.drop(index=list(range(i + 1)))
                    df = rename_duplicate_columns(df)
                    df = df.fillna('')
                    processed_dfs.append(df)
                    break

        if not processed_dfs:
            st.sidebar.error("No usable data found in the uploaded file.")
            return None

        return processed_dfs

    except Exception as e:
        st.sidebar.error(f"Error processing Excel file: {e}")
        return None

# Function to rename duplicate columns
def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    cols = cols.fillna('Unnamed')
    cols = cols.apply(lambda x: x if 'Unnamed' not in x else None)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        cols.iloc[dup_indices] = [f"{dup}_{i}" for i in range(1, len(dup_indices) + 1)]
    df.columns = cols
    return df.dropna(axis=1, how='all')

if __name__ == "__main__":
    main()

