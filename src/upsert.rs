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
use regex::Regex;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about = "Upload folder contents to Pinecone as vector embeddings")]
struct Args {
    /// Path to the folder containing files to process
    #[arg(short, long)]
    folder: String,
    
    /// Maximum size of each chunk in characters (default: 1500)
    #[arg(short = 's', long, default_value = "1500")]
    chunk_size: usize,
    
    /// Overlap between chunks in characters (default: 200)
    #[arg(short = 'o', long, default_value = "200")]
    chunk_overlap: usize,
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
    chunk_index: usize,
    total_chunks: usize,
    chunk_text: String,
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

/// Split text into chunks with a specified size and overlap
fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    // Define pattern for paragraph breaks
    let paragraph_breaks = Regex::new(r"\n\s*\n").unwrap();
    
    while start < text.len() {
        let mut end = std::cmp::min(start + chunk_size, text.len());
        
        // If we're not at the end of the text, try to find a natural break point
        if end < text.len() {
            // Look for paragraph breaks within the chunk_size from the end
            let search_start = if end > 100 { end - 100 } else { start };
            let chunk_text = &text[search_start..end];
            
            if let Some(mat) = paragraph_breaks.find_iter(chunk_text).last() {
                // Found paragraph break, adjust end position
                end = search_start + mat.end();
            } else {
                // No paragraph break found, look for sentence boundaries
                // We'll manually search for periods, exclamation points, and question marks 
                // followed by whitespace
                let end_text = &text[search_start..end];
                let mut found_sentence_end = false;
                
                // Try to find the last sentence end in the search window
                for i in (0..end_text.len()).rev() {
                    // Check if character at position i is a sentence-ending punctuation
                    if i + 1 < end_text.len() && 
                       (end_text.chars().nth(i) == Some('.') || 
                        end_text.chars().nth(i) == Some('!') || 
                        end_text.chars().nth(i) == Some('?')) && 
                       end_text.chars().nth(i + 1) == Some(' ') {
                        // Found a sentence boundary
                        end = search_start + i + 2; // Include the space after punctuation
                        found_sentence_end = true;
                        break;
                    }
                }
                
                // If no sentence boundary found, fall back to word boundary (space)
                if !found_sentence_end {
                    if let Some(last_space) = text[start..end].rfind(' ') {
                        end = start + last_space + 1;
                    }
                }
            }
        }
        
        // Extract the chunk and add to results
        chunks.push(text[start..end].to_string());
        
        // Adjust start for next chunk, considering overlap
        start = if end == text.len() {
            // We've reached the end
            end
        } else {
            // Move back by overlap amount, but ensure we're making forward progress
            std::cmp::max(start + 1, end - chunk_overlap)
        };
    }
    
    chunks
}

async fn process_folder(folder_path: &str, chunk_size: usize, chunk_overlap: usize) -> Result<(), AppError> {
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
            let content_type = get_content_type(&path.to_path_buf());
            
            let relative_path = path.strip_prefix(&folder)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            // For text files, split into chunks
            if content_type == "text" {
                let chunks = chunk_text(&content, chunk_size, chunk_overlap);
                println!("  Split into {} chunks", chunks.len());
                
                // Process each chunk
                for (chunk_index, chunk) in chunks.iter().enumerate() {
                    let embedding = generate_embedding(chunk).await?;
                    
                    // Create a unique ID for each chunk
                    let chunk_id = format!("{}#chunk{}", relative_path, chunk_index);
                    
                    let vector = PineconeVector {
                        id: chunk_id,
                        values: embedding,
                        metadata: PineconeMetadata {
                            filename: filename.clone(),
                            content_type: content_type.clone(),
                            path: relative_path.clone(),
                            chunk_index,
                            total_chunks: chunks.len(),
                            // Store a preview of the chunk text (limited to 500 chars)
                            chunk_text: if chunk.len() > 500 {
                                format!("{}...", &chunk[0..500])
                            } else {
                                chunk.clone()
                            },
                        },
                    };
                    
                    vectors.push(vector);
                    
                    // Batch uploads in chunks of 100 vectors
                    if vectors.len() >= 100 {
                        println!("  Upserting batch of {} vectors to Pinecone...", vectors.len());
                        upsert_to_pinecone(vectors).await?;
                        vectors = Vec::new();
                    }
                }
            } else {
                // For non-text files, just create a single vector
                let embedding = generate_embedding(&content).await?;
                
                let vector = PineconeVector {
                    id: relative_path.clone(),
                    values: embedding,
                    metadata: PineconeMetadata {
                        filename,
                        content_type,
                        path: relative_path,
                        chunk_index: 0,
                        total_chunks: 1,
                        chunk_text: content.clone(),
                    },
                };
                
                vectors.push(vector);
                
                // Batch uploads in chunks of 100 vectors
                if vectors.len() >= 100 {
                    println!("  Upserting batch of {} vectors to Pinecone...", vectors.len());
                    upsert_to_pinecone(vectors).await?;
                    vectors = Vec::new();
                }
            }
        }
    }
    
    // Upload any remaining vectors
    if !vectors.is_empty() {
        println!("  Upserting final batch of {} vectors to Pinecone...", vectors.len());
        upsert_to_pinecone(vectors).await?;
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    
    let args = Args::parse();
    
    println!("Processing folder: {}", args.folder);
    println!("Chunk size: {} characters with {} character overlap", args.chunk_size, args.chunk_overlap);
    
    process_folder(&args.folder, args.chunk_size, args.chunk_overlap)
        .await
        .context("Failed to process folder")?;
    
    println!("Successfully uploaded folder contents to Pinecone!");
    
    Ok(())
}