from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import os
# import tiktoken
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def main():
    print("Hello from llm-rcg24-rag!")

    chunk_size = 1000
    chunk_overlap=100

    load_dotenv()

    llm = HuggingFaceEndpoint(
        repo_id='tiiuae/falcon-7b-instruct',
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Load a the RCG24 document
    loader = PyPDFLoader(os.path.join("input", "rcg24.pdf"))
    document = loader.load()

    # Split the RCG24 document
    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )

    all_splits = rc_splitter.split_documents(document)
    
    print(len(all_splits))
    all_splits = all_splits[0:50]

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Index chunks
    vectorstore = FAISS.from_documents(
        all_splits,
        embedding=embedding_function
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    # print(document[0])

    message = """
    Read the following chunks of text from the OECD Regions and Cities at a Glance 24:
    {rcgtext}
    Now reply the following question based on the text you just read:
    {question}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([("human", message)])

    llm = HuggingFaceEndpoint(
        repo_id='tiiuae/falcon-7b-instruct',
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Split the RCG24 document
    rag_chain = ({"rcgtext": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm)
    
    response = rag_chain.invoke("What is the report about?")
    print(response)
    


if __name__ == "__main__":
    
    main()
