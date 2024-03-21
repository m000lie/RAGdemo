from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


class Core:
    def __init__(self, query, pdf_list):
        self.pdf_list = pdf_list
        self.query = query

    def process_documents(self):
        loader = [PyPDFLoader(x) for x in self.pdf_list]
        all_documents = []
        for loaded in loader:
            pages = loaded.load_and_split()
            all_documents.append(pages[0])
        print(all_documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(all_documents, embeddings)
        retriever = db.as_retriever()

        template = """Answer the question based only on the following context, be sure to keep your answers concise without missing out on any key points:
                {context}

                Question: {question}
                """
        prompt = PromptTemplate.from_template(template)

        self.loader, self.retriever, self.prompt = loader, retriever, prompt

    def load_and_run(self):
        n_gpu_layers = (
            -1
        )  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            max_tokens=512,
            model_path="/home/muirwood/models/mistral.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=8096,
        )
        # model = ChatOpenAI()

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(self.query)
