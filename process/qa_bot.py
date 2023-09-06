from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA


class QA_Bot:
    def __init__(self, explanation):
        self.explanation = explanation

    def store_in_vectordb(self):
        document = Document(page_content=self.explanation)

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents([document])

        vectordb = Chroma.from_documents(
            chunked_documents, embedding=OpenAIEmbeddings(), persist_directory="./data"
        )
        vectordb.persist()

        self.agent = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo"),
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        )

    def retrieve(self, query):
        result = self.agent({"query": query}, return_only_outputs=True)
        result["result"]
