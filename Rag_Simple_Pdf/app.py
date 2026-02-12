import streamlit as st
from dotenv import load_dotenv

from Rag_Simple_Pdf.model_selector import SimpleModelSelector
from Rag_Simple_Pdf.pdf_processor import SimplePDFProcessor
from Rag_Simple_Pdf.rag_system import SimpleRAGSystem


load_dotenv()


def main():
    ############################################################
    # PAGE TITLE
    ############################################################
    st.title("Rag System")

    ############################################################
    # SESSION STATE
    ############################################################
    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    # if "patient_names" not in st.session_state:
    #     st.session_state.patient_names = {}

    ############################################################
    # MODEL SELECTION (SIDEBAR)
    ############################################################

    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    ############################################################
    # MODEL CHANGE HANDLING
    ############################################################
    # Check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()  # Clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None  # Reset RAG system
        # st.warning("Embedding model changed. Please re-upload your documents.")

    ############################################################
    # RAG SYSTEM INITIALIZATION
    ############################################################

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
            if "patient_names" not in st.session_state:
                st.session_state.patient_names = st.session_state.rag_system.get_patient_names()

        # Display current embedding model info
        embedding_info = st.session_state.rag_system.get_embedding_info(model_selector)
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    ############################################################
    # FILE UPLOAD
    ############################################################

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
                    st.session_state.patient_names.append(inferred_name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    ############################################################
    # QUERY INTERFACE
    ############################################################

    doc_count = st.session_state.rag_system.get_document_count()
    if doc_count > 0:
        st.markdown("---")
        st.subheader("Query Your Documents")
        query = st.text_input("Ask a question:")
        patient_names = sorted(set(st.session_state.patient_names))
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
                results = st.session_state.rag_system.query_documents(
                    query, where=where
                )
                if results and results["documents"]:
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        # Display results
                        st.markdown("### Answer:")
                        st.write(response)

                        with st.expander("View Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
    else:
        st.info("Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()
