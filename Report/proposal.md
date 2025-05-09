# DataMorph: Automating Format Detection & Transformation with GenAI

**Group 15:** Sai Prerana Mandalika, Pranav Patil, Zenan Wang

---

Organizations in major domains like finance, healthcare, and e-commerce handle large and diverse datasets in multiple formats (JSON, CSV, XML, Avro, Parquet, etc.). Each format has a unique structure, making data integration and processing challenging. Before data can be analyzed, it must be parsed, cleaned, and converted into a standardized format—a task that is currently done manually or with custom scripts. This approach is time-consuming, inefficient, and difficult to scale, as new datasets often require modifications to existing pipelines.

This project introduces an AI-powered data engineering agent that automates schema inference, data transformation, and serialization using Generative AI (GenAI) and AI agents. The agent follows a four-step approach:

### 1. Detecting Files and Inferring Schema
The system scans an S3 bucket or local storage, detects file formats, and extracts schema details using Retrieval-Augmented Generation (RAG) and libraries like `pandas`, `pyarrow`, and `xmltodict`.

### 2. AI-Generated Code for Transformation
Using LLMs (GPT-4, Claude, Llama 3), the agent dynamically generates Python scripts for cleaning, type standardization, and format conversion.

### 3. Executing the Generated Code
The generated scripts are executed in a controlled environment like Docker, with AI-driven validation to ensure correctness and security.

### 4. Retrieval-Augmented Generation (RAG)
The system maintains a domain-specific knowledge base of data format specifications and schema transformation patterns, used to enrich prompt construction and ensure context-aware generation.

### 5. Storing Transformed Data Efficiently
The processed data is stored in Parquet (for analytics), Avro (for streaming), or JSON/CSV (for interoperability), optimized for performance.

---

Our findings demonstrate that AI-assisted schema detection and transformation significantly reduce manual effort while ensuring high accuracy, scalability, and efficiency. This project paves the way for self-adaptive, automated data engineering workflows, eliminating the need for manual intervention in schema handling and data conversion.
