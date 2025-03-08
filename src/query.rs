use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(author, version, about = "Query OpenAI GPT-4o with context from Pinecone")]
struct Args {
    /// The query to ask
    #[arg(short, long)]
    query: String,
    
    /// Number of top results to fetch from Pinecone (default: 5)
    #[arg(short, long, default_value = "5")]
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

#[derive(Debug, Deserialize)]
struct PineconeMatch {
    id: String,
    score: f32,
    metadata: PineconeMetadata,
}

#[derive(Debug, Deserialize)]
struct PineconeMetadata {
    filename: String,
    content_type: String,
    path: String,
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

async fn fetch_file_content(matches: &[PineconeMatch]) -> Result<String, AppError> {
    let mut context = String::new();
    
    for (i, m) in matches.iter().enumerate() {
        context.push_str(&format!("\n\n--- Document {} (path: {}, score: {:.4}) ---\n", 
            i + 1, m.metadata.path, m.score));
        
        // In a production environment, you would fetch the actual file content here
        // based on the path or ID stored in Pinecone metadata.
        // For this demo version, we'll use a simulated approach:
        
        // If the file exists on the local system, try to read its content
        let path = std::path::Path::new(&m.metadata.path);
        if path.exists() && path.is_file() {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    // For long files, include just the first 1000 characters
                    let content = if content.len() > 1000 {
                        format!("{}... [content truncated]", &content[..1000])
                    } else {
                        content
                    };
                    context.push_str(&content);
                },
                Err(_) => {
                    // If we can't read the file, just use the ID
                    context.push_str(&format!("ID: {}\n", m.id));
                }
            }
        } else {
            // If the file doesn't exist locally, use the ID as fallback
            context.push_str(&format!("ID: {}\n", m.id));
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
        "You are a helpful assistant that answers questions based on the provided context. \
        If the context doesn't contain relevant information, say so instead of making up an answer. \
        Here is the context information extracted from documents:\n\n{}", 
        context
    );
    
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: system_message,
        },
        ChatMessage {
            role: "user".to_string(),
            content: query.to_string(),
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
        println!("{}. {} (score: {:.4})", i + 1, m.metadata.path, m.score);
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