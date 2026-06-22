import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import chromadb
from langchain_community.llms import Ollama


llm = Ollama(model="gemma3:4b")
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📄 PDF Chatbot")

    st.markdown("""
    ### Features

    ✅ PDF Upload
    ✅ ChromaDB
    ✅ Gemini AI
    ✅ Semantic Search
    ✅ RAG Pipeline
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.sidebar.markdown("""
## 👨‍💻 Developer

**Arshit Goyal**

B.Tech Chemical Engineering
IIT Ropar

### 🚀 Tech Stack
- Streamlit
- ChromaDB
- Gemini AI
- Sentence Transformers
- LangChain
""")


st.title("📄 AI PDF Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    text = ""

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    st.sidebar.metric("Chunks", len(chunks))
    st.sidebar.metric("Embeddings", len(embeddings))
    st.sidebar.write("### Current PDF")
    for file in uploaded_files:
        st.sidebar.info(file.name)

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="pdf_chunks"
    )

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )

    st.success("PDF Processed Successfully!")
    st.sidebar.metric("Chunks", len(chunks))
    st.sidebar.metric("Embeddings", len(embeddings))
    st.sidebar.success("PDF Stored")


question = st.chat_input(
    "Ask anything about your PDF..."
)

if question is not None and question.strip() != "":

    with st.chat_message("user"):
        st.write(question)

        query_embedding = model.encode([question])

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=2
        )

        context = "\n\n".join(results["documents"][0])

        prompt = f"""
        You are an AI PDF assistant.

        Use ONLY the provided context.

        If the answer is present, explain it clearly in simple language.

        If the answer is not found in the context, say:
        "Answer not found in the uploaded PDF."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        answer = None

        try:
            with st.spinner("🤖 Thinking..."):
                answer = llm.invoke(prompt)
            with st.chat_message("assistant"):
                st.write(answer)
            with st.expander("📚 View Retrieved Sources"):
                for i, doc in enumerate(results["documents"][0]):
                    st.markdown(f"### Source {i+1}")
                    st.write(doc)

        except Exception as e:
            st.error(f"Ollama Error: {e}")

            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            if answer:
                with st.chat_message("assistant"):
                    st.write(answer)

            if answer:
                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

            with st.expander("📚 View Retrieved Sources"):
                for i, doc in enumerate(results["documents"][0]):
                    st.write(f"### Source {i+1}")
                    st.write(doc)

                    st.divider()

if st.session_state.messages:

    st.markdown("## 💬 Chat History")

    for msg in st.session_state.messages:

        with st.chat_message("user"):
            st.write(msg["question"])

        with st.chat_message("assistant"):
            st.write(msg["answer"])

    st.divider()
