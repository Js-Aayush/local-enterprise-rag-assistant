from pypdf import PdfReader
pdf_path = "ppe_manual.pdf"
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text +"\n"

print(full_text[:2000])
def split_chunk(text,chunk_size=1000,overlap =200):
    chunks = []
    start = 0
    while start<len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = split_chunk(
    full_text,
    chunk_size=1000,
    overlap = 200
)
print("Total chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])