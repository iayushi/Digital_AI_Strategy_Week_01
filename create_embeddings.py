#!/usr/bin/env python3
"""
Vector Embeddings Helper Script for Digital AI Strategy Course

This script helps create vector embeddings from course content (.md or .txt files)
and stores them in a ChromaDB database for use with the RAG application.

Author: Digital AI Strategy Course
Compatible with: DAIS_Week1_RAG.py
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Union

# Required imports for ChromaDB compatibility
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch


class EmbeddingCreator:
    """Helper class for creating vector embeddings from course content."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_gpu: bool = False,
        persist_directory: str = "./course_vectordb"
    ):
        """
        Initialize the EmbeddingCreator.
        
        Args:
            model_name (str): HuggingFace model name for embeddings
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
            use_gpu (bool): Whether to use GPU for embedding generation
            persist_directory (str): Directory to store the vector database
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize the embedding model
        self._init_embedding_model()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print(f"✓ Initialized EmbeddingCreator")
        print(f"  Model: {model_name}")
        print(f"  Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        print(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"  Database location: {persist_directory}")
    
    def _init_embedding_model(self):
        """Initialize the HuggingFace embedding model."""
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            print(f"Initializing embedding model on {device}...")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print(f"✓ Successfully initialized {self.model_name}")
            
        except Exception as e:
            print(f"✗ Error initializing embedding model: {e}")
            print("Falling back to CPU...")
            self.use_gpu = False
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def load_file(self, file_path: Union[str, Path]) -> str:
        """
        Load content from a text or markdown file.
        
        Args:
            file_path (Union[str, Path]): Path to the file
            
        Returns:
            str: File content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.txt', '.md']:
            raise ValueError(f"Unsupported file type: {file_path.suffix}. Only .txt and .md files are supported.")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✓ Loaded {file_path} ({len(content)} characters)")
            return content
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                print(f"✓ Loaded {file_path} with latin-1 encoding ({len(content)} characters)")
                return content
            except Exception as e:
                raise ValueError(f"Could not read file {file_path}: {e}")
    
    def create_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Create Document objects from multiple files.
        
        Args:
            file_paths (List[Union[str, Path]]): List of file paths
            
        Returns:
            List[Document]: List of Document objects with metadata
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                content = self.load_file(file_path)
                file_path = Path(file_path)
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create Document objects with metadata
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    
                    doc = Document(page_content=chunk, metadata=metadata)
                    all_documents.append(doc)
                
                print(f"✓ Created {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
                continue
        
        print(f"✓ Total documents created: {len(all_documents)}")
        return all_documents
    
    def create_or_update_vectorstore(self, documents: List[Document], collection_name: str = "course_materials") -> Chroma:
        """
        Create a new vector database or update an existing one.
        
        Args:
            documents (List[Document]): Documents to add to the database
            collection_name (str): Name of the collection
            
        Returns:
            Chroma: The vector store instance
        """
        try:
            # Check if database already exists
            if os.path.exists(self.persist_directory):
                print(f"Existing database found at {self.persist_directory}")
                print("Loading existing database...")
                
                # Load existing vectorstore
                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name=collection_name
                )
                
                # Add new documents
                print(f"Adding {len(documents)} new documents...")
                vectorstore.add_documents(documents)
                
            else:
                print(f"Creating new database at {self.persist_directory}")
                
                # Create new vectorstore
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=collection_name
                )
            
            # Persist changes
            print("Persisting database to disk...")
            vectorstore.persist()
            
            print(f"✓ Vector database successfully created/updated")
            print(f"  Location: {self.persist_directory}")
            print(f"  Collection: {collection_name}")
            print(f"  Total documents: {vectorstore._collection.count()}")
            
            return vectorstore
            
        except Exception as e:
            print(f"✗ Error creating/updating vector database: {e}")
            raise
    
    def test_vectorstore(self, vectorstore: Chroma, test_query: str = "digital strategy") -> None:
        """
        Test the vector database with a sample query.
        
        Args:
            vectorstore (Chroma): The vector store to test
            test_query (str): Query to test with
        """
        try:
            print(f"\nTesting vector database with query: '{test_query}'")
            
            # Perform similarity search
            results = vectorstore.similarity_search(test_query, k=3)
            
            print(f"✓ Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                content_preview = doc.page_content[:100].replace('\n', ' ')
                source = doc.metadata.get('filename', 'Unknown')
                print(f"  {i}. {source}: {content_preview}...")
            
        except Exception as e:
            print(f"✗ Error testing vector database: {e}")


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Create vector embeddings from course content files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings from single file
  python create_embeddings.py course_material.md
  
  # Process multiple files
  python create_embeddings.py file1.md file2.txt file3.md
  
  # Use custom chunk size and GPU
  python create_embeddings.py --chunk-size 1000 --use-gpu files/*.md
  
  # Specify custom database location
  python create_embeddings.py --db-path ./my_vectordb files/*.txt
  
  # Use different embedding model
  python create_embeddings.py --model sentence-transformers/all-mpnet-base-v2 *.md
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='One or more .md or .txt files to process'
    )
    
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help='HuggingFace model name for embeddings (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Size of text chunks for processing (default: 500)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between text chunks (default: 50)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for embedding generation if available'
    )
    
    parser.add_argument(
        '--db-path',
        default='./course_vectordb',
        help='Directory to store the vector database (default: ./course_vectordb)'
    )
    
    parser.add_argument(
        '--collection-name',
        default='course_materials',
        help='Name of the vector database collection (default: course_materials)'
    )
    
    parser.add_argument(
        '--test-query',
        default='digital strategy',
        help='Query to test the database with (default: "digital strategy")'
    )
    
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip testing the vector database after creation'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("Vector Embeddings Creator for Digital AI Strategy Course")
        print("=" * 60)
        
        # Validate input files
        valid_files = []
        for file_path in args.files:
            path = Path(file_path)
            if path.exists() and path.suffix.lower() in ['.txt', '.md']:
                valid_files.append(file_path)
            else:
                print(f"⚠ Skipping invalid file: {file_path}")
        
        if not valid_files:
            print("✗ No valid files found. Please provide .txt or .md files.")
            return 1
        
        print(f"Processing {len(valid_files)} files...")
        
        # Initialize embedding creator
        creator = EmbeddingCreator(
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_gpu=args.use_gpu,
            persist_directory=args.db_path
        )
        
        # Create documents
        print("\nProcessing files...")
        documents = creator.create_documents(valid_files)
        
        if not documents:
            print("✗ No documents were created. Please check your files.")
            return 1
        
        # Create or update vector database
        print("\nCreating/updating vector database...")
        vectorstore = creator.create_or_update_vectorstore(
            documents, 
            collection_name=args.collection_name
        )
        
        # Test the database
        if not args.no_test:
            creator.test_vectorstore(vectorstore, args.test_query)
        
        print("\n" + "=" * 60)
        print("✓ Vector embeddings created successfully!")
        print(f"  Database location: {args.db_path}")
        print(f"  To use with DAIS_Week1_RAG.py, update PERSIST_DIRECTORY to: '{args.db_path}'")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if '--debug' in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())