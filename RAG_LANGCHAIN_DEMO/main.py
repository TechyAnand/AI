import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Set your Gemini API Key (from Google AI Studio)
os.environ["GOOGLE_API_KEY"] = " "; 


# Step 1: Load documents
loader = TextLoader("sample_data.txt")
documents = loader.load()

# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Embed the chunks
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(chunks, embedding)

# Step 4: Set up retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # or models/gemini-1.5-flash

# Step 5: Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 6: Ask a question
query = [ "What is the name of the village where Elias lived?", "Why did Elias refuse the King’s request?", "What is Elias's Secret?"]

for i in query:
   print (20*"-")
   print("Question to AI: " + i )
   response = qa_chain.invoke(i)
   print(response)
   print (20*"-")

