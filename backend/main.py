from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader

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
    print(extracted_text)

    return {"filename": file.filename, "size": len(contents)}

# function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text