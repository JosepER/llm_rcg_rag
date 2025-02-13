from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import os
# import tiktoken
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def main():
    print("Hello from llm-rcg24-rag!")

    chunk_size = 500
    chunk_overlap=50

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
    
    # print(len(all_splits))
    # all_splits = all_splits[0:50]

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Index chunks
    vectorstore = Milvus.from_documents(
        documents=all_splits,
        embedding=embedding_function,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="rcg24"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    message = """
    Read the following chunks of text from the OECD Regions and Cities at a Glance 24:
    {rcgtext}
    Now reply the following question based on the text you just read:
    {question}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([("human", message)])

    llm = HuggingFaceEndpoint(
        repo_id='deepseek-ai/DeepSeek-R1',
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Split the RCG24 document
    rag_chain = ({"rcgtext": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm)
    
    print("Enter your question related to RCG24:")
    user_question = input()

    # user_question =  """What is difference between the region
    #     with the highest and lowest real GDP per capita growth in 2015-2022 period?"""
    
    response = rag_chain.invoke(user_question)

    print("\n")
    print("--------------------------------------------------")
    print(response)
    

if __name__ == "__main__":
    
    main()
