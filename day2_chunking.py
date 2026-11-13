from pypdf import PdfReader

# 1. PDF file ko load karke saara text nikalna
reader = PdfReader("sample.pdf")
full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

print(f"Total pages: {len(reader.pages)}")
print(f"Total characters extracted: {len(full_text)}")

# 2. Text ko overlapping Chunks mein todne ka function
def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = create_chunks(full_text)

print(f"\nTotal Chunks bane: {len(chunks)}")
print("\n--- Pehla Chunk (Sample) ---")
print(chunks[0])