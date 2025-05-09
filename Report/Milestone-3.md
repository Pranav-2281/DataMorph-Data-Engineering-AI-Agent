# Milestone 3: Agents and Evaluation

This milestone focused on the integration and evaluation of intelligent agents within the GenAI-powered Streamlit application. The goal was to design and implement modular, interpretable agents using LangGraph and LangChain frameworks, and to evaluate their effectiveness across real-world data transformation workflows.

---

## 1. Application Overview

The project’s `main.py` and `agents.py` files together compose a fully functional Streamlit web application for dynamic, step-based data processing. Users can upload data files in various formats and enter transformation instructions in natural language.

Upon receiving the file and instruction, the system uses an LLM to interpret the request and dynamically routes the task through a series of well-defined agents.

---

## 2. Agent Workflow

Each major stage of the ETL pipeline is handled by a dedicated LangGraph agent:
- **Deserialization Agent**: Detects file format and loads data into memory
- **Schema Inference Agent**: Analyzes and outputs a schema based on the input data
- **Transformation Agent**: Applies user-defined transformations using context-aware LLM prompts and RAG support
- **Serialization Agent**: Converts the transformed data into the target output format
- **Format Conversion Agent**: Converts files between formats (e.g., CSV to Parquet)
- **S3 Upload Agent**: Uploads the final output to AWS S3 for persistent storage

All agents are encapsulated with timing logic to monitor performance and debug latency bottlenecks.

---

## 3. RAG-Enhanced Processing

The transformation agent is empowered by a Retrieval-Augmented Generation (RAG) system that retrieves transformation-specific context (e.g., formatting guidelines, cleaning rules) from a domain knowledge base indexed via ChromaDB. This context is used to enrich the prompts passed to the LLM, enabling more accurate and domain-aware code generation.

---

## 4. Dynamic Task Routing

Based on the user input, LangGraph dynamically routes control through the appropriate agents. Intermediate outputs (such as schema, cleaned data previews, or errors) are displayed in real time on the Streamlit interface, enhancing transparency and traceability.

---

## 5. Evaluation Metrics

To assess the functionality and performance of each agent, the following metrics were recorded:
- **Latency per agent**: Time taken for each step (e.g., deserialization, transformation)
- **Transformation accuracy**: Output schema correctness and data integrity
- **Pipeline robustness**: Ability to handle nested or inconsistent formats
- **Success rate**: Number of complete, correct runs across different test cases

These evaluations informed refinements in prompt structure, agent orchestration, and error handling logic.

This milestone marks the successful operationalization of GenAI agents for end-to-end data engineering workflows.

