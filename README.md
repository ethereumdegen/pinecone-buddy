# Pinecone Buddy

A simple command-line tool that processes files in a directory and uploads them to Pinecone as vector embeddings.

## Features

### Upload Features
- Recursively process all files in a specified directory
- Respects .gitignore patterns and other git ignore rules (skips ignored files)
- Generate vector embeddings using OpenAI's embedding models
- Automatically resizes embeddings to match your Pinecone index dimensions
- Upload embeddings to Pinecone in batches
- Store metadata with each vector (filename, content type, relative path)
- Intelligent text chunking for better search results
- Smart chunk boundary detection using natural text breaks
- Configurable chunk size and overlap parameters
- Proper metadata for chunks including original file path and chunk information

### Query Features
- Smart content retrieval with parallel processing for better performance
- Content caching to avoid repeated file reads
- Proper handling of different file types
- Intelligent base directory detection for relative paths
- Enhanced context formatting for better AI responses
- Query tracing with unique IDs and timestamps for auditing
- Production-ready error handling and recovery
- Advanced prompt engineering for better context utilization

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pinecone_buddy.git
   cd pinecone_buddy
   ```

2. Make sure you have Rust installed. If not, install it from [https://rustup.rs/](https://rustup.rs/).

3. Create a `.env` file based on the `.env.example` template:
   ```
   cp .env.example .env
   ```

4. Edit the `.env` file with your OpenAI and Pinecone credentials.

## Usage

### Uploading content to Pinecone

To upload files from a folder to Pinecone:

```
cargo run --bin upsert -- -f /path/to/your/folder
```

You can customize the chunking behavior with additional options:

```
cargo run --bin upsert -- --folder /path/to/your/folder --chunk-size 2000 --chunk-overlap 250
```

Options:
- `-f, --folder`: Path to the folder containing files to process (required)
- `-s, --chunk-size`: Maximum size of each chunk in characters (default: 1500)
- `-o, --chunk-overlap`: Overlap between chunks in characters (default: 200)

### Querying with context from Pinecone

To query OpenAI GPT-4o with context from documents stored in Pinecone:

```
cargo run --bin query -- -q "Your question here?"
```

or

```
cargo run --bin query -- --query "Your question here?" --top-k 10
```


 



```
cargo run --bin query -- --query "tell me about the advantages of vector embeddings "  
```








The `--top-k` parameter (default: 5) controls how many similar documents to retrieve from Pinecone.

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_EMBEDDING_MODEL` (optional): The embedding model to use for vector embeddings (defaults to text-embedding-ada-002)
- `OPENAI_CHAT_MODEL` (optional): The chat model to use for querying (defaults to gpt-4o)
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX`: Your Pinecone index name
- `PINECONE_HOST`: Your Pinecone host URL (e.g., https://your-index-name-project-id.svc.your-region.pinecone.io)
- `PINECONE_DIMENSION`: The dimension of your Pinecone index (e.g., 1024, 1536)

### Embedding Models and Dimensions

Different OpenAI embedding models produce vectors with different dimensions:
- text-embedding-ada-002: 1536 dimensions
- text-embedding-3-small: 1536 dimensions
- text-embedding-3-large: 3072 dimensions

Make sure your `PINECONE_DIMENSION` matches your Pinecone index's dimension. If there's a mismatch, the embeddings will be automatically resized (truncated or padded) to match.

## Building

```
cargo build --release
```

The compiled binary will be in `target/release/upsert`.

## License

MIT