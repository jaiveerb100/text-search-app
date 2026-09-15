from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running!"}

# endpoint to upload a PDF file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    extracted_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(extracted_text)
    embeddings = [get_embedding(chunk) for chunk in chunks]


    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding has {len(embeddings[0])} dimensions")

    return {"filename": file.filename, "size": len(contents)}

# function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# function to create chunks of text
def chunk_text(text, chunk_size = 150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding