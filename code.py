import os
import re
import docx
import fitz
import cohere
from dotenv import load_dotenv

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.document_compressors import LLMChainFilter

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
cohere_api_key = "ZNfhw5DTkPxmRK9vNON7gc4z56Q1WQ2ctNkMdUl3"
co = cohere.Client(cohere_api_key)

# ✅ Text extraction functions
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path, file_type):
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")

# ✅ Load and chunk documents
text = extract_text("DBMS notes.docx", "docx")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])

# ✅ Vector store setup
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.5})

# ✅ Load LLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Compression & reranker
compressor = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# ✅ Prompt template (clean, no leakage)
prompt = PromptTemplate(
    template="""
Answer the following question based ONLY on the provided context.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ✅ Cohere reranking
def rerank_docs(docs, query, top_n=4):
    candidates = [doc.page_content for doc in docs]
    response = co.rerank(query=query, documents=candidates, top_n=top_n)
    return [docs[result.index] for result in response.results]

# ✅ Runnable chain structure
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

question="what is acid properties in DBMS"

# 1. Retrieve relevant chunks (no compression filter)
retrieved_docs = retriever.invoke(question)

# 2. Rerank with Cohere (optional but recommended)
reranked_docs = rerank_docs(retrieved_docs, question)

# 3. Format context
context_text = "\n\n".join(doc.page_content for doc in reranked_docs)

# 4. Prepare final prompt
final_prompt = prompt.format(context=context_text, question=question)

# 5. Generate full, free-form answer
answer = llm.invoke(final_prompt)

print("\n===== Final Answer =====\n")
print(answer)

