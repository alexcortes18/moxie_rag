import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


class SimpleRAGSystem:
    """Simple RAG implementation."""

    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chromadb")

        # Setup embedding function based on model
        self.setup_embedding_function()

        # Setup LLM
        if llm_model == "openai":
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Setup the appropriate embedding function."""
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
            else:  # chroma default
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def setup_collection(self):
        """Setup collection with proper dimension handling."""
        collection_name = f"documents_{self.embedding_model}"
        try:
            try:
                # Collection loading if exists
                collection = self.db.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                )
                st.success(
                    f"Using existing collection for {self.embedding_model} embeddings"
                )
            except Exception:
                # Create collection if collection didn't exist before
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": self.embedding_model},
                )
                st.success(
                    f"Created new collection for {self.embedding_model} embeddings"
                )
        except Exception as e:
            return f"Error at setting up collection: {str(e)}"
        return collection

    def add_documents(self, chunks):
        """Add documents to ChromaDB."""
        try:
            # Ensure collection exists
            if not self.collection:
                self.collection = self.setup_collection()
            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
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
        """Query documents and return relevant chunks."""
        try:
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
            return results
        except Exception as e:
            return f"Error querying documents: {str(e)}"

    def generate_response(self, query, context):
        """Generate response using LLM."""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self, model_selector):
        """Get information about current embedding model."""
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }

    def get_document_count(self):
        """Return number of documents in the current collection."""
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            return self.collection.count()
        except Exception as e:
            st.error(f"Error getting document count: {str(e)}")
            return 0
