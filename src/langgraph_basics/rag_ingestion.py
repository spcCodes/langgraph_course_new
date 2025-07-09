from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader=DirectoryLoader("data",glob="./*.txt",loader_cls=TextLoader)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function = len
)
new_docs = text_splitter.split_documents(documents=docs)
doc_strings = [doc.page_content for doc in new_docs]

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Save the vectorstore with a particular name, e.g., "bangladesh_economy"
db = Chroma.from_documents(new_docs, embeddings, persist_directory="bangladesh_economy")
retriever = db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    print("\n \n Data ingestion complete")