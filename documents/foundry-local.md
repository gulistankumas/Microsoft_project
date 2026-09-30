# Foundry Local

Foundry Local runs AI models directly on your device using ONNX Runtime.
After the models and runtime components have been downloaded, inference runs
locally. An Azure subscription and a cloud API key are not required for this demo.

The Foundry Local SDK supports Python, C#, JavaScript, and Rust. It provides
model discovery, downloads, loading, inference, and unloading. On Windows,
the WinML package can manage hardware-specific execution providers.

# Retrieval-augmented generation

Retrieval-augmented generation, or RAG, retrieves relevant information from
your documents before asking a language model to answer a question. RAG does
not retrain the language model. Instead, it adds retrieved text to the prompt.

Long documents are split into overlapping chunks. An embedding model turns
each chunk into a numerical vector. The question is also embedded, and cosine
similarity ranks the chunks. The most similar chunks become the answer context.

This project uses qwen3-embedding-0.6b for embeddings and qwen2.5-0.5b for
answers. The vector index stays in memory and is rebuilt when the app starts.
Source filenames, chunk numbers, and PDF page numbers help users check answers.

# Limitations

Retrieval scores measure vector similarity, not the probability that an answer
is correct. Language models can still produce incorrect answers and citations.
Users should check the displayed sources. Scanned PDFs require OCR, which this
application does not provide.

Reference: https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app
