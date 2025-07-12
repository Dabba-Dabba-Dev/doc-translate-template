from docx import Document

doc = Document('ocr/enhanced_processed_document.docx')
extracted_lines = []

# Process each paragraph and its runs
for para in doc.paragraphs:
    line = ""
    for run in para.runs:
        # Directly add the run text. If a run's text contains multiple spaces,
        # they will be preserved if they’re present in the underlying XML.
        line += run.text
    extracted_lines.append(line)

# Join lines with newlines (or process them further as needed)
output_text = "\n".join(extracted_lines)
print(output_text)
