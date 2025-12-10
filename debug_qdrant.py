
import asyncio
import os
import sys
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Configuration
HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = "semantic_memory"
MODEL_NAME = "all-MiniLM-L6-v2"

async def main():
    print(f"Connecting to Qdrant at {HOST}:{PORT}...")
    client = AsyncQdrantClient(url=f"http://{HOST}:{PORT}")
    
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    try:
        # 1. Check Collection
        try:
            info = await client.get_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' exists. Points: {info.points_count}")
        except Exception as e:
            print(f"Collection check failed: {e}")
            return

        # 2. Insert Memory
        text = "The quick brown fox jumps over the lazy dog."
        print(f"\nEncoding: '{text}'")
        vector = model.encode(text).tolist()
        
        point_id = str(uuid.uuid4())
        await client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={"content": text, "type": "debug_semantic"}
            )]
        )
        print(f"Inserted point {point_id}")

        # 3. Search Memory
        query_text = "fox dog"
        print(f"\nSearching for: '{query_text}'")
        query_vector = model.encode(query_text).tolist()
        
        try:
            response = await client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.0
            )
            
            print(f"Points found: {len(response.points)}")
            for pt in response.points:
                content = pt.payload.get('content', '')
                print(f" - Score: {pt.score:.4f} | Content: {content[:50]}...")
                
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
