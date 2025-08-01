#!/usr/bin/env python3
"""
Test Qdrant connection only
"""

import asyncio
import os
from qdrant_client import AsyncQdrantClient

async def test_connection():
    print("🧪 Testing Qdrant Connection")
    print("=" * 50)
    
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    
    print(f"\nConnecting to Qdrant at {host}:{port}...")
    
    try:
        # Use HTTP URL for local development
        url = f"http://{host}:{port}"
        client = AsyncQdrantClient(url=url, prefer_grpc=False)
        
        # Test connection
        collections = await client.get_collections()
        print(f"✅ Connected successfully!")
        print(f"📦 Found {len(collections.collections)} collections:")
        
        for collection in collections.collections:
            print(f"  - {collection.name}")
        
        # Check if documents collection exists
        try:
            info = await client.get_collection("documents")
            print(f"\n📄 'documents' collection exists:")
            print(f"  - Points: {info.points_count}")
            print(f"  - Status: {info.status}")
        except:
            print("\n📄 'documents' collection does not exist yet")
        
        await client.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
