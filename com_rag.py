from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



load_dotenv()



loader = PyPDFLoader("mydocument.pdf")

documents = loader.load()

print("Number of pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))



embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)



vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector store created successfully.")



retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)



def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



parser = StrOutputParser()


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)


print("\n====================================")
print("        PDF RAG CHATBOT")
print("====================================")
print("Ask questions about your PDF.")
print("Type 'exit' to stop.")
print()


while True:

    question = input("You: ")

    # Stop chatbot
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Generate answer
    answer = rag_chain.invoke(question)

    print("\nAI:", answer)
    print()