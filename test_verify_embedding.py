#!/usr/bin/env python3
"""
Verify document embedding results
"""

import asyncio
import json
from qdrant_client import AsyncQdrantClient

async def verify_embedding():
    print("🧪 Verifying Document Embedding Results")
    print("=" * 50)
    
    # Connect to Qdrant
    client = AsyncQdrantClient(url="http://localhost:6333", prefer_grpc=False)
    
    try:
        # Get collection info
        info = await client.get_collection("documents")
        print(f"\n📄 Documents collection:")
        print(f"  - Total chunks: {info.points_count}")
        print(f"  - Status: {info.status}")
        
        # Get a sample of documents
        result = await client.scroll(
            collection_name="documents",
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = result
        
        print(f"\n📋 Sample chunks:")
        for i, point in enumerate(points, 1):
            print(f"\n  Chunk {i}:")
            print(f"    - ID: {point.id}")
            print(f"    - Source: {point.payload.get('source_file', 'unknown')}")
            print(f"    - Chunk Index: {point.payload.get('chunk_index', 0)}")
            print(f"    - Total Chunks: {point.payload.get('total_chunks', 0)}")
            content = point.payload.get('content', '')
            print(f"    - Content Preview: {content[:100]}...")
        
        # Test search
        print(f"\n🔍 Testing search functionality...")
        
        # Direct vector search (using dummy vector for testing)
        search_results = await client.search(
            collection_name="documents",
            query_vector=[0.1] * 384,  # Dummy vector
            limit=3,
            with_payload=True
        )
        
        print(f"\n✅ Search works! Found {len(search_results)} results")
        
        await client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        await client.close()

if __name__ == "__main__":
    asyncio.run(verify_embedding())
