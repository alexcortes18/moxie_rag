import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import uuid
import hashlib

load_dotenv()

# CONSTANTS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class SimpleModelSelector:
    def __init__(self):
        # Available LLM models
        self.llm_models = {"openai": "GPT-4", "ollama": "Llama3"}
        
        # Available embedding models with their dimensions
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
        }
    
    def select_models(self):
        """ Let the user select models through Streamlit UI"""
        st.sidebar.title("📚 Model Selection")
        
        #Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options = list(self.llm_models.keys()),
            format_func= lambda x: self.llm_models[x]
        )
        
        embedding = st.sidebar.radio(
            "Choose Embedding Model",
            options = list(self.embedding_models.keys()),
            format_func= lambda x: self.embedding_models[x]["name"]
        )
        return llm, embedding
 
class SimplePDFProcessor:
    """Handle PDF processing and chunking"""
    
    def __init__(self, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def readPDF(self, pdf_file):
        """Read PDF and extract the text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def compute_file_hash(self, pdf_file):
        """Stable hash of PDF bytes to detect duplicates across sessions."""
        content = pdf_file.getvalue()
        return hashlib.sha256(content).hexdigest()

    def infer_patient_name(self, text, pdf_file):
        """Best-effort patient name inference from PDF text or filename."""
        import re

        candidates = []
        patterns = [
            r"Patient\s*Name\s*[:\-]\s*(.+)",
            r"Name\s*[:\-]\s*(.+)",
            r"Patient\s*[:\-]\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                raw = raw.split("\n")[0].strip()
                raw = re.sub(r"\s{2,}", " ", raw)
                if 3 <= len(raw) <= 80:
                    candidates.append(raw)
        if candidates:
            return candidates[0]

        # Fallback to filename without extension
        base = os.path.splitext(pdf_file.name)[0]
        base = base.replace("_", " ").replace("-", " ").strip()
        return base or "Unknown"
    
    def create_chunks(self, text, pdf_file, patient_name=None, file_hash=None):
        """Split text into chunks"""
        chunks = []
        start = 0
       
        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size
           
            # If not at the start, include overlap
            if start > 0:
               start = start - self.chunk_overlap
            
            # Get Chunk
            chunk = text[start:end]
            
            #Try to end the chunk at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "source": pdf_file.name,
                        "patient_name": patient_name or "Unknown",
                        "file_hash": file_hash,
                    }
                }
            )
            start = end
        return chunks
    
class SimpleRAGSystem:
    """Simple RAG implementation"""
    
    def __init__(self, embedding_model = "openai", llm_model = "openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        #Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chromadb")
        
        # Setup embedding function based on model
        self.setup_embedding_function()
        
        # Setup LLM
        if llm_model == "openai":
            self.llm = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
        else:
            self.llm = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key= "ollama"
            )
        
        self.collection = self.setup_collection()
        
    def setup_embedding_function(self):
        """Setup the appropriate embedding function"""
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small",
                )
            elif self.embedding_model == "nomic":
                # For Nomic embeddings via Ollama
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key="ollama",
                    api_base="http://localhost:11434/v1",
                    model_name="nomic-embed-text",
                )

                # Alternative if needed:
                # self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                #     model_name="all-MiniLM-L6-v2"
                # )
            else:  # chroma default
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e
    
    def setup_collection(self):
        """Setup collection with proper dimension handling"""
        collection_name = f"documents_{self.embedding_model}"
        try:
            try:
                #Collection loading if exists
                collection = self.db.get_collection(
                    name = collection_name,
                    embedding_function= self.embedding_fn,
                )
                st.success(
                    f"Using existing collection for {self.embedding_model} embeddings"
                )
            except Exception as e:
                # Create collection if collection didn't exist before
                collection = self.db.create_collection(
                    name = collection_name,
                    embedding_function = self.embedding_fn,
                    metadata= {"model": self.embedding_model},
                )
                st.success(
                    f"Created new collection for {self.embedding_model} embeddings"
                )
        except Exception as e:
            return f"Error at setting up collection: {str(e)}"        
        return collection
    
    def add_documents(self, chunks):
        """Add documents to ChromaDb"""
        try:
            #Ensure collection exists:
            if not self.collection:
                self.collection = self.setup_collection()
            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
    
    def file_hash_exists(self, file_hash):
        """Check if a file hash is already stored in the collection."""
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            results = self.collection.get(where={"file_hash": file_hash}, limit=1)
            return bool(results and results.get("ids"))
        except Exception as e:
            st.error(f"Error checking file hash: {str(e)}")
            return False
        
    def query_documents(self, query, n_results=3, where=None):
        """Query documents and return relevant chunks"""
        try:
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(
                query_texts = [query],
                n_results = n_results,
                where = where,
            )
            return results
        except Exception as e:
            return f"Error querying documents: {str(e)}"
        
    def generate_response(self, query, context):
        """Generate response using LLM"""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """
            response = self.llm.chat.completions.create(
                model = "gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
    
    def get_embedding_info(self):
        """Get information about current embedding model"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }
            
# =========================================================================================

def main():
    st.title("🤖 Rag System")
    
    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "patient_map" not in st.session_state:
        st.session_state.patient_map = {}
    
    # Initialize model selector
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()
    
    # Check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()  # Clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None  # Reset RAG system
        st.warning("Embedding model changed. Please re-upload your documents.")
        
    # Initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
            
        # Display current embedding model info
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return
    
    # File Upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf_file:
        # Process PDF
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                # Check for duplicates by file hash (always)
                file_hash = processor.compute_file_hash(pdf_file)
                if st.session_state.rag_system.file_hash_exists(file_hash):
                    st.warning(
                        f"Already ingested: **{pdf_file.name}** (embedding: **{embedding_model}**). Skipping."
                    )
                    st.session_state.processed_files.add(pdf_file.name)
                    return

                # Extract text
                text = processor.readPDF(pdf_file)
                # Infer patient name and create chunks
                inferred_name = processor.infer_patient_name(text, pdf_file)
                st.info(f"Inferred patient name: {inferred_name}")
                chunks = processor.create_chunks(
                    text, pdf_file, inferred_name, file_hash=file_hash
                )
                # Add to database
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.session_state.patient_map[pdf_file.name] = inferred_name
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")
        query = st.text_input("Ask a question:")
        patient_names = sorted(set(st.session_state.patient_map.values()))
        patient_filter = st.selectbox(
            "Filter by patient (optional)",
            options=["All patients"] + patient_names,
        )

        if query:
            with st.spinner("Generating response..."):
                # Get relevant chunks
                where = None
                if patient_filter != "All patients":
                    where = {"patient_name": patient_filter}
                results = st.session_state.rag_system.query_documents(query, where=where)
                if results and results["documents"]:
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        # Display results
                        st.markdown("### 📝 Answer:")
                        st.write(response)

                        with st.expander("View Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
    else:
        st.info("👆 Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()
