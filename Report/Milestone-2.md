# Milestone 2: Data Preparation and RAG Implementation

In this milestone, the focus is on two primary objectives: preparing diverse input datasets and implementing a Retrieval-Augmented Generation (RAG) system to enhance the performance of the transformation agent.

---

## 1. Data Preparation

The data preparation phase involves collecting, organizing, and validating datasets in a variety of formats, including:
- CSV
- JSON
- Excel

These formats were selected to evaluate the system's capabilities in schema inference, transformation logic, and output serialization across structurally diverse inputs. Each dataset was structured to include potential edge cases such as missing values, inconsistent types, and nested records to stress-test the agents.

---

## 2. Vector Store and Knowledge Base Setup

To support RAG, a semantic search system was configured using Chroma (or a similar technology). This vector store indexes domain-specific documents that contain:
- Best practices for data cleaning and formatting
- Code snippets for transformation tasks
- Schema examples and structural conventions

These documents were embedded into vector representations and loaded at application startup for efficient retrieval.

---

## 3. RAG Pipeline Integration

The RAG pipeline is embedded within the `transformation_agent`. It retrieves relevant documentation based on the user’s natural language instruction and appends that context to the prompt before sending it to the LLM. The flow involves:
- Parsing the user’s instruction
- Matching it with relevant documents from the vector store
- Constructing a structured prompt that includes: 
  - The original instruction
  - Retrieved knowledge
  - A data preview from the uploaded file

The resulting prompt is passed to an LLM (e.g., GPT-4), which returns a transformation script tailored to the file and instruction context.

---

## 4. Validation and Integration

The RAG-enhanced agent is validated using benchmark instructions on known datasets. Outputs are assessed for:
- Schema alignment
- Data integrity
- Execution correctness

Once validated, the transformation agent is integrated into the LangGraph pipeline, where it functions in sequence with other agents such as deserialization, serialization, and cloud upload. The integration ensures that RAG-powered transformations are compatible with the rest of the data workflow.
