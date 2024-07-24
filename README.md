# Diabetes Management Assistant Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist users in managing diabetes effectively. The chatbot provides personalized advice, tips, and resources on various aspects of diabetes management, including diet, exercise, medication, and lifestyle changes.

## Features

- Utilizes the Zephyr-7b-beta language model for generating responses
- Implements RAG to enhance responses with relevant information from a diabetes management knowledge base
- Uses sentence transformers and FAISS for efficient document retrieval
- Provides a user-friendly chat interface built with Gradio

## Technical Stack

- Python
- Gradio for the web interface
- HuggingFace's InferenceClient for the language model
- PyMuPDF for PDF processing
- Sentence-Transformers for text embedding
- FAISS for vector similarity search
- NumPy for numerical operations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MannatSidhu-4371127/Customized-LLM-APP/.git
   cd Customized-LLM-APP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the knowledge base PDF file (`DB_management.pdf`) in the project directory.

2. Run the application:
   ```
   python app.py
   ```

3. Open the provided URL in your web browser to interact with the chatbot.

## How It Works

1. The app loads and processes the PDF knowledge base on startup.
2. It builds a vector database from the extracted text using sentence transformers.
3. When a user asks a question, the app searches for relevant information in the vector database.
4. The retrieved information is combined with the user's query and sent to the Zephyr-7b-beta model.
5. The model generates a response based on the combined input, providing diabetes management advice.

## Customization-

You can customize the chatbot by:
- Replacing the knowledge base PDF with your own diabetes management resources
- Adjusting the system message to focus on specific aspects of diabetes management
- Modifying the response generation parameters (max_tokens, temperature, top_p)

## Example Queries used-

- "What foods should I avoid to keep my blood sugar stable?"
- "What are the best exercises for someone with diabetes?"
- "How can I manage stress to help control my diabetes?"

## Acknowledgements-

- This project uses the [Zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model from HuggingFace.
- The diabetes management knowledge base is derived from "MANAGEMENT OF DIABETES MELLITUS: STANDARDS OF CARE AND CLINICAL PRACTICE GUIDELINES" edited by Dr A.A.S. Alwan.
- Vector similarity search is powered by FAISS.

---

For any questions or contributions, please open an issue or submit a pull request.
