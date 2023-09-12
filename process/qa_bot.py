import logging

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Q&A")


class QA_Bot:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        self.llm = ChatOpenAI(
            temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=openai_key
        )
        self.ddg_search = DuckDuckGoSearchRun()
        self.agent = None

    def store_in_vectordb(self, explanation):
        document = Document(page_content=explanation)

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents([document])
        logger.info("Documents ready")

        vectordb = Chroma.from_documents(
            chunked_documents,
            embedding=OpenAIEmbeddings(openai_api_key=self.openai_key),
            persist_directory="./data",
        )
        vectordb.persist()
        logger.info("Documents inserted to vectordb")

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say 'False'.

        {context}

        Question: {question}"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.agent = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
        )
        logger.info("Agent ready!!")

    def summarize_result(self, db_result, internet_result):
        response_prompt = f"""
                                Given 2 explanation from internet and a custom Vector database. Prepare a answer by understanding them
                                in 20 words unless length of answer not specified.\n\n

                                Internet : {internet_result}\n\n

                                Vector DB : {db_result}
                                """
        final_response = self.llm.predict(response_prompt)
        return final_response

    def retrieve(self, query):
        db_result = self.agent({"query": query}, return_only_outputs=True)
        if db_result["result"] == "False":
            internet_result = self.ddg_search.run(query)
            final_response = self.summarize_result(db_result["result"], internet_result)
        else:
            final_response = db_result["result"]

        logger.info("Result ready!")
        return final_response
