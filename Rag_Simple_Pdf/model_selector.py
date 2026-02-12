import streamlit as st


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
        """Let the user select models through Streamlit UI."""
        st.sidebar.title("Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        embedding = st.sidebar.radio(
            "Choose Embedding Model",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )
        return llm, embedding
