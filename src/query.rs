use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about = "Query OpenAI GPT-4o with context from Pinecone")]
struct Args {
    /// The query to ask
    #[arg(short, long)]
    query: String,
    
    /// Number of top results to fetch from Pinecone (default: 10)
    #[arg(short, long, default_value = "10")]
    top_k: usize,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("Environment variable not found: {0}")]
    EnvVar(String),
    
    #[error("OpenAI API error: {0}")]
    OpenAI(String),
    
    #[error("Pinecone API error: {0}")]
    Pinecone(String),
    
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("File content error: {0}")]
    FileContent(String),
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct PineconeQueryRequest {
    namespace: String,
    vector: Vec<f32>,
    top_k: usize,
    include_metadata: bool,
}

#[derive(Debug, Deserialize)]
struct PineconeQueryResponse {
    matches: Vec<PineconeMatch>,
}

#[derive(Debug, Deserialize, Clone)]
struct PineconeMatch {
    id: String,
    score: f32,
    metadata: PineconeMetadata,
}

#[derive(Debug, Deserialize, Clone)]
struct PineconeMetadata {
    filename: String,
    content_type: String,
    path: String,
    // New fields for chunking support
    chunk_index: Option<usize>,
    total_chunks: Option<usize>,
    chunk_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

async fn generate_embedding(query: &str) -> Result<Vec<f32>, AppError> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::EnvVar("OPENAI_API_KEY".to_string()))?;
    
    // Get the embedding model to use from environment or default to ada-002
    let model = env::var("OPENAI_EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-ada-002".to_string());
    
    let client = Client::new();
    
    let request = OpenAIEmbeddingRequest {
        model,
        input: vec![query.to_string()],
    };
    
    let response = client.post("https://api.openai.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::OpenAI(format!("API request failed: {}", error_text)));
    }
    
    let embedding_response: OpenAIEmbeddingResponse = response.json()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if let Some(first_embedding) = embedding_response.data.first() {
        let mut embedding = first_embedding.embedding.clone();
        
        // Check if we need to resize the embedding to match Pinecone dimensions
        if let Ok(target_dim) = env::var("PINECONE_DIMENSION").map(|s| s.parse::<usize>()) {
            if let Ok(target_dim) = target_dim {
                if embedding.len() != target_dim {
                    println!("Resizing embedding from {} to {} dimensions", embedding.len(), target_dim);
                    resize_embedding(&mut embedding, target_dim);
                }
            }
        }
        
        Ok(embedding)
    } else {
        Err(AppError::OpenAI("No embedding returned".to_string()))
    }
}

/// Resize embedding vector to target dimension
fn resize_embedding(embedding: &mut Vec<f32>, target_dim: usize) {
    match embedding.len().cmp(&target_dim) {
        std::cmp::Ordering::Less => {
            // If embedding is smaller than target, pad with zeros
            embedding.resize(target_dim, 0.0);
        }
        std::cmp::Ordering::Greater => {
            // If embedding is larger than target, truncate
            embedding.truncate(target_dim);
        }
        std::cmp::Ordering::Equal => {
            // Already the right size, do nothing
        }
    }
}

async fn query_pinecone(embedding: Vec<f32>, top_k: usize) -> Result<Vec<PineconeMatch>, AppError> {
    let api_key = env::var("PINECONE_API_KEY")
        .map_err(|_| AppError::EnvVar("PINECONE_API_KEY".to_string()))?;
    
    let _index_name = env::var("PINECONE_INDEX")
        .map_err(|_| AppError::EnvVar("PINECONE_INDEX".to_string()))?;
    
    let pinecone_host = env::var("PINECONE_HOST")
        .map_err(|_| AppError::EnvVar("PINECONE_HOST".to_string()))?;
    
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| AppError::Request(e))?;
    
    let request = PineconeQueryRequest {
        namespace: "default".to_string(),
        vector: embedding,
        top_k,
        include_metadata: true,
    };
    
    let url = format!("{}/query", pinecone_host);
    
    let response = client.post(&url)
        .header("Api-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::Pinecone(format!("Failed to query vectors: {}", error_text)));
    }
    
    let query_response: PineconeQueryResponse = response.json()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    Ok(query_response.matches)
}

/// A cache for storing file contents to avoid repeated disk reads
struct ContentCache {
    // Maps file paths to their contents
    cache: HashMap<String, String>,
    // Base directory for relative paths
    base_dir: Option<PathBuf>,
    // Maximum size for text content before truncation (in characters)
    max_text_size: usize,
}

impl ContentCache {
    fn new(max_text_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            base_dir: None,
            max_text_size,
        }
    }
    
    fn with_base_dir(mut self, base_dir: Option<PathBuf>) -> Self {
        self.base_dir = base_dir;
        self
    }
    
    /// Get content from cache or read from disk
    async fn get_content(&mut self, path_str: &str) -> Result<String, AppError> {
        // Check if content is already in cache
        if let Some(content) = self.cache.get(path_str) {
            return Ok(content.clone());
        }
        
        // Determine the full path
        let path = if let Some(base_dir) = &self.base_dir {
            // If we have a base directory, interpret path as relative
            let rel_path = Path::new(path_str);
            if rel_path.is_absolute() {
                rel_path.to_path_buf()
            } else {
                base_dir.join(rel_path)
            }
        } else {
            // Otherwise, use the path as is
            Path::new(path_str).to_path_buf()
        };
        
        // Check if path exists and is a file
        if !path.exists() {
            return Err(AppError::FileContent(format!("File not found: {}", path_str)));
        }
        
        if !path.is_file() {
            return Err(AppError::FileContent(format!("Not a file: {}", path_str)));
        }
        
        // Get file extension to determine content type
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        // Read and process the file based on its type
        let content = match extension.as_str() {
            // Text files
            "txt" | "md" | "rs" | "js" | "py" | "html" | "css" | "json" | "toml" | "yaml" | "yml" | "c" | "cpp" | "h" | "go" | "java" | "kt" | "ts" | "sh" | "bash" | "rb" => {
                self.read_text_file(&path).await?
            },
            // Binary files or other types
            _ => {
                format!("[Binary file or unsupported format: {}]", path.display())
            }
        };
        
        // Cache the content
        self.cache.insert(path_str.to_string(), content.clone());
        
        Ok(content)
    }
    
    /// Read and process text file
    async fn read_text_file(&self, path: &Path) -> Result<String, AppError> {
        let content = fs::read_to_string(path)
            .map_err(|e| AppError::Io(e))?;
        
        // Truncate content if necessary
        if content.len() > self.max_text_size {
            Ok(format!("{}... [content truncated, showing {}/{} characters]", 
                &content[0..self.max_text_size], 
                self.max_text_size, 
                content.len()))
        } else {
            Ok(content)
        }
    }
}

/// Determine the base directory for content if possible
fn determine_base_dir(matches: &[PineconeMatch]) -> Option<PathBuf> {
    // Strategy: Look for common path prefixes among multiple files
    if matches.is_empty() {
        return None;
    }
    
    // Get all paths and convert to PathBuf
    let paths: Vec<PathBuf> = matches.iter()
        .map(|m| Path::new(&m.metadata.path).to_path_buf())
        .collect();
    
    // If we only have one path, use its parent directory
    if paths.len() == 1 {
        return paths[0].parent().map(|p| p.to_path_buf());
    }
    
    // Find current working directory as fallback
    match std::env::current_dir() {
        Ok(cwd) => {
            // Check if all paths are within the current directory
            let all_in_cwd = paths.iter()
                .all(|p| p.starts_with(&cwd));
            
            if all_in_cwd {
                return Some(cwd);
            }
        },
        Err(_) => {}
    }
    
    // If no common base found, return None
    None
}

async fn fetch_file_content(matches: &[PineconeMatch]) -> Result<String, AppError> {
    // Exit early if no matches
    if matches.is_empty() {
        return Ok(String::from("No matching documents found."));
    }
    
    let mut context = String::new();
    
    // Try to determine a base directory for the paths
    let base_dir = determine_base_dir(matches);
    
    // Create a content cache with a 5000 character limit for text files
    let content_cache = Arc::new(Mutex::new(
        ContentCache::new(5000).with_base_dir(base_dir)
    ));
    
    // Include metadata about the matches
    context.push_str(&format!("Found {} matching documents:\n\n", matches.len()));
    
    // Use a vector to collect all content results
    let mut content_futures = Vec::new();
    
    // Create futures for content retrieval
    for (i, m) in matches.iter().enumerate() {
        let m_clone = m.clone();
        let cache = content_cache.clone();
        
        // Create a future for retrieving this content
        let content_future = tokio::spawn(async move {
            // Build a more detailed header with chunk info if available
            let document_header = if let (Some(chunk_index), Some(total_chunks)) = 
                (m_clone.metadata.chunk_index, m_clone.metadata.total_chunks) {
                format!("--- Document {} (path: {}, chunk: {}/{}, score: {:.4}) ---\n", 
                    i + 1, m_clone.metadata.path, chunk_index + 1, total_chunks, m_clone.score)
            } else {
                format!("--- Document {} (path: {}, score: {:.4}) ---\n", 
                    i + 1, m_clone.metadata.path, m_clone.score)
            };
            
            // If we have chunk_text stored in metadata, use that directly
            if let Some(chunk_text) = &m_clone.metadata.chunk_text {
                return format!("\n\n{}\n{}\n", document_header, chunk_text);
            }
            
            // Otherwise, try to get content from cache or disk
            let content_result = cache.lock().await.get_content(&m_clone.metadata.path).await;
            
            match content_result {
                Ok(content) => {
                    format!("\n\n{}\n{}\n", document_header, content)
                },
                Err(e) => {
                    format!("\n\n{}\nError retrieving content: {}\nID: {}\n", 
                        document_header, e, m_clone.id)
                }
            }
        });
        
        content_futures.push(content_future);
    }
    
    // Wait for all content retrievals to complete
    for future in content_futures {
        match future.await {
            Ok(result) => {
                context.push_str(&result);
            },
            Err(e) => {
                context.push_str(&format!("\n\nError processing document: {}\n", e));
            }
        }
    }
    
    Ok(context)
}

async fn query_openai(query: &str, context: &str) -> Result<String, AppError> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::EnvVar("OPENAI_API_KEY".to_string()))?;
    
    let model = env::var("OPENAI_CHAT_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    
    let client = Client::new();
    
    let system_message = format!(
        "You are a helpful, accurate assistant that answers questions based on the provided context. \
        You'll be given a set of document excerpts that may contain information relevant to the user's query. \
        Follow these guidelines carefully:\
        \n\n1. Base your response ONLY on the provided document excerpts. \
        \n2. If the context doesn't contain sufficient information to answer the question fully, \
        acknowledge the limitations and explain what's missing. \
        \n3. If you're unsure or the context is ambiguous, indicate your uncertainty. \
        \n4. Always cite your sources by referring to the document numbers (e.g., 'According to Document 3...'). \
        \n5. Never make up information or pretend to know things not contained in the provided documents. \
        \n6. Be concise and focus on providing accurate, factual responses. \
        \n\nHere is the context information extracted from documents:\n\n{}", 
        context
    );

    println!("attaching pinecone context {}", context);
    
    // Add a trace_id for query tracking
    let trace_id = uuid::Uuid::new_v4().to_string();
    println!("Query trace ID: {}", trace_id);
    
    // Add a timestamp to the request
    let timestamp = chrono::Utc::now().to_rfc3339();
    
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: system_message,
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!("[trace_id: {}][timestamp: {}] {}", trace_id, timestamp, query),
        },
    ];
    
    let request = OpenAIChatRequest {
        model,
        messages,
    };
    
    let response = client.post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::OpenAI(format!("API request failed: {}", error_text)));
    }
    
    let chat_response: OpenAIChatResponse = response.json()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if let Some(choice) = chat_response.choices.first() {
        Ok(choice.message.content.clone())
    } else {
        Err(AppError::OpenAI("No response returned".to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    
    let args = Args::parse();
    
    println!("Processing query: {}", args.query);
    
    // 1. Generate embedding for the query
    let embedding = generate_embedding(&args.query)
        .await
        .context("Failed to generate embedding")?;
    
    // 2. Query Pinecone for similar vectors
    let matches = query_pinecone(embedding, args.top_k)
        .await
        .context("Failed to query Pinecone")?;
    
    if matches.is_empty() {
        println!("No matching documents found in Pinecone.");
        return Ok(());
    }
    
    println!("Found {} matching documents:", matches.len());
    for (i, m) in matches.iter().enumerate() {
        if let (Some(chunk_index), Some(total_chunks)) = (m.metadata.chunk_index, m.metadata.total_chunks) {
            println!("{}. {} (chunk: {}/{}, score: {:.4})", 
                i + 1, m.metadata.path, chunk_index + 1, total_chunks, m.score);
        } else {
            println!("{}. {} (score: {:.4})", i + 1, m.metadata.path, m.score);
        }
    }
    
    // 3. Fetch content of matching documents
    let context = fetch_file_content(&matches)
        .await
        .context("Failed to fetch file content")?;
    
    // 4. Query OpenAI with the original query and context
    let response = query_openai(&args.query, &context)
        .await
        .context("Failed to query OpenAI")?;
    
    println!("\nAI Response:");
    println!("{}", response);
    
    Ok(())
}