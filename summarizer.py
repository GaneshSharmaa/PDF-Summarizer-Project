# importing the required modules
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

# loading the environment variables
load_dotenv()

# initializing the llm
llm = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct", temperature = 0.5)

# creating a prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You're a document summarizer, whatever document (text, PDF, website URL) you get, you summarize it.
        There are some instructions that you need to follow:
        - Never add anything or never add made up fact into the summary.
        - Always keep the summary to the point.
        - The language or style of summary should match the original document."""
    ),
    (
        "human",
        "{docs}"
    )
])

def summarize(document_docs: str):
    prompt = prompt_template.format_messages(
        docs = document_docs
    )

    response = llm.invoke(prompt)
    return response.content


def process_url(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    return summarize(docs[0].page_content)

def process_text(file_path: str):
    loader = TextLoader(file_path, encoding = "utf-8")
    docs = loader.load()
    return summarize(docs[0].page_content)

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    formatted_docs = []

    for doc in docs:
        formatted_docs.append(doc.page_content)
    full_text = "\n".join(formatted_docs)
    return summarize(full_text)