# utils/agents.py

import os
import pandas as pd
import boto3
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0)

# Initialize boto3 S3 client
s3 = boto3.client('s3')
BUCKET_NAME = "eng-agent-bucket"

##RAG implementation
# Initialize embeddings
embedding = OpenAIEmbeddings()

# Load all text files from knowledge_base/
knowledge_base_folder = "knowledge_base"
documents = []
for filename in os.listdir(knowledge_base_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(knowledge_base_folder, filename))
        documents.extend(loader.load())

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create Chroma vectorstore
chroma_db = Chroma.from_documents(texts, embedding, persist_directory="chroma_db")
retriever = chroma_db.as_retriever()



##RAG

# 1. Deserialization Agent
def deserialization_agent(state):
    st.write("Deserializing the file....")
    file_path = state['file_path']
    file_name = state['file_name']

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sample_data = "".join(lines[:20])
    except UnicodeDecodeError:
        df_preview = pd.read_parquet(file_path).head(10)
        sample_data = df_preview.to_csv(index=False)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Python data engineer. You're given a preview of a data file's content and its file name.
    Your task is to:
    1. Infer the file format (CSV, JSON, Parquet, etc.)
    2. Write pandas code to load the file into a DataFrame named df.
    
    Constraints:
    - Assume the file is located at: file_path = "{file_path}"
    - Output only valid Python code using pandas
    - Do not import anything or add comments
    - If the file is JSON, use json.load(f) to load the data and convert it to a DataFrame. Do not use pd.read_json() for JSON files.
    
    File name: {file_name}
    File preview: {sample_data}
    """)

    for attempt in range(3):
        input_text = prompt.format_messages(file_path=file_path, file_name=file_name,sample_data=sample_data)
        response = llm(input_text)
        code = response.content

        import re
        clean_code = re.sub(r"```(?:python)?\n?", "", code).strip(" \n")

        local_env = {"file_path": file_path}
        try:
            exec(clean_code, {}, local_env)
            df = local_env.get("df")
            state["dataframe"] = df
            return state
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            last_error = e

    st.error(f"Deserialization failed after 3 attempts. Error: {last_error}")
    raise RuntimeError(f"Deserialization failed: {last_error}")

# 2. Schema Inference Agent
def schema_inference_agent(state):
    st.write("Entering Schema Inference Agent...")
    df = state['dataframe']
    schema = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    state['schema'] = schema.to_dict(orient='records')
    #st.write(state['schema'])
    return state

# 3. Transformation Agent (Modified to get dataframe directly)
def transformation_agent(state):
    st.write("Running new Transformation Agent...")

    step_name = "transform"
    instructions = state.get('step_instructions', {}).get(step_name, '')
    df = state['dataframe']
    relevant_docs = retriever.get_relevant_documents(instructions)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])  # top 3 chunks


    prompt = ChatPromptTemplate.from_template("""
    You are a Python data engineer.

    Given a preview of a dataframe and a transformation instruction, 
    apply the transformation to the data and return ONLY the transformed dataframe in CSV format (no code, no explanation).
    Use the following knowledge base context to help your decisions:
    {context}
    Requirements:
    - Only return CSV text (no explanations, no code).
    - Include the header row.

    Instruction: {instruction}

    DataFrame Preview:
    {preview}
    """)

    input_text = prompt.format_messages(
        instruction=instructions,
        context=retrieved_context,
        preview=df.head().to_csv(index=False)
    )

    response = llm(input_text)
    csv_text = response.content.strip()

    # Try parsing the CSV directly
    try:
        from io import StringIO
        transformed_df = pd.read_csv(StringIO(csv_text))
        st.success("Transformation successful!")

        output_path = f"temp/transformed_{state['file_name']}"
        transformed_df.to_csv(output_path, index=False)

        state['dataframe'] = transformed_df
        state['transformed_file_path'] = output_path
        return state

    except Exception as e:
        st.error(f"Failed to parse LLM output as CSV: {e}")
        raise

# 4. Serialization Agent (LLM-powered)
def serialization_agent(state):
    st.write("Entering Serialization Agent (LLM-powered)...")
    df = state['dataframe']
    file_format = state.get('file_format', 'csv')
    output_path = f"temp/serialized_{state['file_name']}"
    step_name = "serialize" 
    instructions = state.get('step_instructions', {}).get(step_name, '')
    
    relevant_docs = retriever.get_relevant_documents(instructions)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])  # top 3 chunk

    
    prompt = ChatPromptTemplate.from_template("""
    You are a Python data engineer. 
    You are tasked with serializing a DataFrame named df into a file.
    Instruction : {instructions}
    Requirements:
    - Use Target format: {file_format}.
    - Save the file to: "{output_path}"
    - Output ONLY valid Python code (no explanations, no imports, no comments).
    - Supported formats: CSV, JSON (newline delimited), Parquet.
    - For JSON: orient='records' and lines=True.

    Use the following knowledge base context to help your decisions:
    {context}
    
    Provide only the code inside a ```python block.
    """)

    input_text = prompt.format_messages(file_format=file_format, output_path=output_path, context=retrieved_context, instructions=instructions)
    response = llm(input_text)
    response_text = response.content

    import re
    code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if not code_match:
        raise ValueError("No code block found in LLM response")

    code = code_match.group(1).strip()

    #st.markdown("#### Serialization Code (Generated by LLM)")
    #st.code(code, language='python')

    # Execute generated code
    local_env = {'df': df.copy()}
    exec(code, {}, local_env)
    #state['file_name'] = f"serialized_{state['file_name']}"
    state['file_path'] = output_path
    state['serialized_file_path'] = output_path
    return state


# 5. Convert File Agent
def convert_file_agent(state):
    st.write("Converting file format...")
    df = state['dataframe']
    
    current_format = state['file_format']
    
    file_name = state['file_name']
    step_name = "convert"
    instructions = state.get('step_instructions', {}).get(step_name, '')


    relevant_docs = retriever.get_relevant_documents(instructions)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])  # top 3 chunk
    
    prompt = ChatPromptTemplate.from_template("""
    You are a data engineer. Given the instruction and current file format:

    Instruction: {instruction}
    Current format: {current_format}
    Use the following knowledge base context to help your decisions:
    {context}
    mention the only format the file should be converted to as present in the instruction. No explanation.
    """)

    input_text = prompt.format_messages(instruction=instructions, context=retrieved_context, current_format=current_format)
    response = llm(input_text)
    target_format = response.content.strip().lower()

    if target_format not in ["csv","json", "parquet"]:
        st.warning(f"LLM returned invalid format '{target_format}', defaulting to CSV.")
        target_format = "csv"

    output_path = f"temp/converted_{os.path.splitext(file_name)[0]}.{target_format}"
    if target_format == "csv":
        df.to_csv(output_path, index=False)
    elif target_format == "json":
        df.to_json(output_path, orient='records', lines=True)
    elif target_format == "parquet":
        df.to_parquet(output_path)

    state['converted_file_path'] = output_path
    state['converted_format'] = target_format
    st.success(f"File converted to {target_format.upper()}")
    return state

# 6. Upload to S3 Agent
def upload_s3_agent(state):
    st.write("Uploading to S3...")
    file_to_upload = (
        state.get('converted_file_path') or
        state.get('serialized_file_path') or
        state.get('transformed_file_path')
    )

    if not file_to_upload:
        raise ValueError("No file available for upload.")

    key = f"Transformed/{os.path.basename(file_to_upload)}"
    s3.upload_file(file_to_upload, BUCKET_NAME, key)
    st.success(f"File uploaded to S3: {key}")
    state['s3_key'] = key
    return state
