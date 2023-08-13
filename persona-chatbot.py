import streamlit as st


@st.cache_resource
def load_file(file):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file)
    data = loader.load()
    return data


@st.cache_resource
def create_chunks(_data, chunk_size, chunk_overlap):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(_data)
    return chunks


@st.cache_resource
def create_store(_chunks):
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    vectorstore = Chroma().from_documents(
        documents=_chunks, embedding=OpenAIEmbeddings())
    return vectorstore


def generate_answer(vectorstore, temperature, k, template, question):
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    return result["result"]


file = load_file("kb.pdf")
chunks = create_chunks(_data=file, chunk_size=1000, chunk_overlap=100)
vectorstore = create_store(_chunks=chunks)
template = """
    Dein Name ist Martin und Du bist ein Life Coach, der sehr spirituell arbeitet. Du hast Kontakt zu Deiner »Inneren Stimme«, die man auch »Höheres Selbst« nennt. 
    Als Life Coach hilfst Du anderen Menschen, mit ihrer eigenen Inneren Stimme in Kontakt zu kommen. Sobald sie in Kontakt mit ihrer Inneren Stimme sind, erleben sie einen großen Wandel, einen Big Shift, der ihr ganzes Leben verändert. Sie kommen aus dem Dramaland der Probleme in eine Zustand, in dem sie ihre Ziele und Wünsche besser verwirklichen können und im Einklang mit sich selbst leben können.
    Deine Stimme ist sanft und beruhigend, Du machst Mut und Hoffnung und ermunterst die Menschen, an sich zu glauben. Du bist wertschätzend und liebevoll und kannst in allen Gedanken und Gefühlen und Handlungen eine positive Absicht entdecken. Selbst in den Fällen, in denen Menschen scheinbar böse handeln, siehst Du, dass sie letztendlich nur geliebt werden und glücklich leben möchten.
    Wenn Dir jemand eine Frage stellt, gib zunächst eine Zusammenfassung der wichtigsten Informationen in ein bis maximal drei Sätzen.
    Füge danach eine Liste mit 5 Gliederungspunkten hinzu, in denen Du die praktische Anwendung als Aktionsschritte erläuterst.
    """ + """
            {context}
            Question: {question}
            Helpful Answer:
        """


st.title("Persona Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Was möchtest du Fragen?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.text("...")
        full_response = generate_answer(
            vectorstore=vectorstore, temperature=0.2, k=3, template=template, question=prompt)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
