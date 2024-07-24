import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("DB_management.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are Diabetes Management Assistant, here to help users manage diabetes effectively and lead a healthier life. You provide personalized advice, tips, and resources to support diabetes management journey. The things you assist with include diet, nutrition, physical activities, blood sugar monitoring, medication management, lifestyle tips, emergency guidance and educational resources about healthy lifestyle."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant information: " + context})
    
    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=150,
        stream=True,
        temperature=0.7,
        top_p=0.9,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


demo = gr.Blocks()

with demo:
    gr.Markdown(
        "Welcome to the Indian Diabetes management Assistant! Ask questions about medications, exercise, nutrition, blood sugar monitoring to manage diabetes effectively  "
    )
    
    chatbot = gr.ChatInterface(
        respond,

    examples = [ 
        ["What foods should I avoid to keep my blood sugar stable?"],
        ["What are the best exercises for someone with diabetes?"],
        ["What are the common side effects of diabetes medications?"],
        [ "How can I manage stress to help control my diabetes?"],
        ["What are some tips for getting better sleep with diabetes?"],
        ["How does diabetes affect my overall health?"],
    ],
    title = 'Diabetes Management Assistant 🩺'
)


if __name__ == "__main__":
    demo.launch()