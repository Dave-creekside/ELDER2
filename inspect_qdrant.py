#!/usr/bin/env python3
"""
Inspect Qdrant collection to see what's actually stored
"""

import asyncio
import json
import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def inspect_collection(collection_name="semantic_memory"):
    """Inspect all points in a Qdrant collection"""
    
    # Connect to Qdrant
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    
    client = AsyncQdrantClient(url=f"http://{host}:{port}", prefer_grpc=False)
    
    try:
        # Get collection info
        info = await client.get_collection(collection_name)
        print(f"\n=== Collection: {collection_name} ===")
        print(f"Points count: {info.points_count}")
        print(f"Vectors count: {info.vectors_count}")
        print(f"Status: {info.status}")
        
        # Scroll through all points to get their data
        print(f"\n=== Inspecting all {info.points_count} points ===\n")
        
        offset = None
        point_num = 0
        
        while True:
            # Get batch of points
            result = await client.scroll(
                collection_name=collection_name,
                limit=10,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need the actual vectors
            )
            
            points, next_offset = result
            
            if not points:
                break
                
            for point in points:
                point_num += 1
                print(f"Point #{point_num} (ID: {point.id})")
                print(f"  Payload fields: {list(point.payload.keys())}")
                
                # Show content if it exists
                if 'content' in point.payload:
                    content = point.payload['content']
                    if len(content) > 100:
                        print(f"  Content: {content[:100]}...")
                    else:
                        print(f"  Content: {content}")
                
                # Show other metadata
                for key, value in point.payload.items():
                    if key != 'content':
                        if isinstance(value, str) and len(str(value)) > 100:
                            print(f"  {key}: {str(value)[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                
                print()
            
            if next_offset is None:
                break
            offset = next_offset
        
        # Check if these are document chunks
        print("\n=== Document Analysis ===")
        
        # Try to find points with source_file metadata
        doc_points = 0
        memory_points = 0
        
        offset = None
        while True:
            result = await client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            for point in points:
                if 'source_file' in point.payload:
                    doc_points += 1
                else:
                    memory_points += 1
            
            if next_offset is None:
                break
            offset = next_offset
        
        print(f"Document chunks (with source_file): {doc_points}")
        print(f"Regular memories (no source_file): {memory_points}")
        print(f"Total points: {doc_points + memory_points}")
        
        # Try a sample search to see what's retrievable
        if info.points_count > 0:
            print("\n=== Sample Search Test ===")
            print("Searching with a dummy vector to see what's retrievable...")
            
            # Create a dummy vector (assuming 384 dimensions)
            dummy_vector = [0.1] * 384
            
            search_results = await client.search(
                collection_name=collection_name,
                query_vector=dummy_vector,
                limit=3,
                with_payload=True
            )
            
            print(f"Found {len(search_results)} results in search")
            for i, result in enumerate(search_results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result.score}")
                print(f"  ID: {result.id}")
                if 'content' in result.payload:
                    content = result.payload['content']
                    if len(content) > 100:
                        print(f"  Content preview: {content[:100]}...")
                    else:
                        print(f"  Content: {content}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        env_path = "streamlined_consciousness/.env"
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
    except ImportError:
        pass
    
    asyncio.run(inspect_collection())
