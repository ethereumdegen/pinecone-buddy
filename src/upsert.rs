use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use ignore::WalkBuilder;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(author, version, about = "Upload folder contents to Pinecone as vector embeddings")]
struct Args {
    /// Path to the folder containing files to process
    #[arg(short, long)]
    folder: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
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
struct PineconeVector {
    id: String,
    values: Vec<f32>,
    metadata: PineconeMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
struct PineconeMetadata {
    filename: String,
    content_type: String,
    path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PineconeUpsertRequest {
    vectors: Vec<PineconeVector>,
    namespace: String,
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

async fn generate_embedding(content: &str) -> Result<Vec<f32>, AppError> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::EnvVar("OPENAI_API_KEY".to_string()))?;
    
    // Get the embedding model to use from environment or default to ada-002
    // text-embedding-ada-002 produces 1536 dimensions
    // text-embedding-3-small produces 1536 dimensions
    // text-embedding-3-large produces 3072 dimensions
    let model = env::var("OPENAI_EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());
    
    let client = Client::new();
    
    let request = OpenAIEmbeddingRequest {
        model,
        input: vec![content.to_string()],
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

async fn upsert_to_pinecone(vectors: Vec<PineconeVector>) -> Result<(), AppError> {
    let api_key = env::var("PINECONE_API_KEY")
        .map_err(|_| AppError::EnvVar("PINECONE_API_KEY".to_string()))?;
    
   // let _index_name = env::var("PINECONE_INDEX")
    //    .map_err(|_| AppError::EnvVar("PINECONE_INDEX".to_string()))?;
    
    let pinecone_host = env::var("PINECONE_HOST")
        .map_err(|_| AppError::EnvVar("PINECONE_HOST".to_string()))?;
    
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| AppError::Request(e))?;
    
    let request = PineconeUpsertRequest {
        vectors,
        namespace: "default".to_string(),
    };
    
    let url = format!("{}/vectors/upsert", pinecone_host);
    
    let response = client.post(&url)
        .header("Api-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::Request(e))?;
    
    if response.status().is_success() {
        Ok(())
    } else {
        let error_text = response.text().await.unwrap_or_default();
        Err(AppError::Pinecone(format!("Failed to upsert vectors: {}", error_text)))
    }
}

fn get_content_type(path: &PathBuf) -> String {
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    match extension {
        "txt" | "md" | "rs" | "js" | "py" | "html" | "css" | "json" | "toml" | "yaml" | "yml" => "text".to_string(),
        "jpg" | "jpeg" | "png" | "gif" | "svg" => "image".to_string(),
        "pdf" => "pdf".to_string(),
        _ => "unknown".to_string(),
    }
}

fn read_file_content(path: &PathBuf) -> Result<String, AppError> {
    let content_type = get_content_type(path);
    
    if content_type == "text" {
        fs::read_to_string(path).map_err(AppError::Io)
    } else {
        // For non-text files, just return the filename as content
        Ok(path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string())
    }
}

async fn process_folder(folder_path: &str) -> Result<(), AppError> {
    let folder = PathBuf::from(folder_path);
    
    if !folder.exists() || !folder.is_dir() {
        return Err(AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Folder not found: {}", folder_path),
        )));
    }
    
    let mut vectors = Vec::new();
    
    // Use WalkBuilder to respect .gitignore patterns
    let walker = WalkBuilder::new(&folder)
        .standard_filters(true)  // Use standard filters (includes .gitignore)
        .hidden(true)            // Skip hidden files
        .git_global(true)        // Use global git ignore rules
        .git_ignore(true)        // Use .gitignore rules
        .git_exclude(true)       // Use .git/info/exclude rules
        .build();
    
    for result in walker {
        let entry = match result {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!("Error walking directory: {}", err);
                continue;
            }
        };
        
        let path = entry.path();
        
        if path.is_file() {
            println!("Processing file: {}", path.display());
            
            let content = read_file_content(&path.to_path_buf())?;
            let embedding = generate_embedding(&content).await?;
            
            let relative_path = path.strip_prefix(&folder)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            let vector = PineconeVector {
                id: relative_path.clone(),
                values: embedding,
                metadata: PineconeMetadata {
                    filename,
                    content_type: get_content_type(&path.to_path_buf()),
                    path: relative_path,
                },
            };
            
            vectors.push(vector);
            
            // Batch uploads in chunks of 100 vectors
            if vectors.len() >= 100 {
                upsert_to_pinecone(vectors).await?;
                vectors = Vec::new();
            }
        }
    }
    
    // Upload any remaining vectors
    if !vectors.is_empty() {
        upsert_to_pinecone(vectors).await?;
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    
    let args = Args::parse();
    
    println!("Processing folder: {}", args.folder);
    
    process_folder(&args.folder)
        .await
        .context("Failed to process folder")?;
    
    println!("Successfully uploaded folder contents to Pinecone!");
    
    Ok(())
}