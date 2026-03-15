from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        if not pages:
            raise ValueError("PDF contains no readable text.")

        return pages

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")