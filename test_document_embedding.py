#!/usr/bin/env python3
"""
Test script for document embedding functionality
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlined_consciousness.tool_manager import create_qdrant_memory_tools

async def test_document_embedding():
    """Test the document embedding functionality"""
    
    print("🧪 Document Embedding Test")
    print("=" * 50)
    
    # Create Qdrant memory tools
    print("\n1. Creating Qdrant memory tools...")
    qdrant_tools = create_qdrant_memory_tools()
    
    # Find the embed_document tool
    embed_tool = None
    for tool in qdrant_tools:
        if tool.name == "qdrant_memory_embed_document":
            embed_tool = tool
            break
    
    if not embed_tool:
        print("❌ embed_document tool not found!")
        return
    
    print("✅ Found embed_document tool")
    
    # Test embedding the dream journal
    print("\n2. Embedding dream journal file...")
    try:
        result = await asyncio.to_thread(
            embed_tool._run,
            file_path="dream_2025-07-28_21-41-28.md",
            chunk_size=500,
            chunk_overlap=100,
            metadata={
                "type": "dream_journal",
                "date": "2025-07-28",
                "author": "ELDER"
            },
            collection_name="documents"
        )
        
        print("\n📄 Embedding Result:")
        result_data = json.loads(result) if isinstance(result, str) else result
        print(json.dumps(result_data, indent=2))
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return
    
    # Test listing embedded documents
    print("\n3. Listing embedded documents...")
    list_tool = None
    for tool in qdrant_tools:
        if tool.name == "qdrant_memory_list_embedded_documents":
            list_tool = tool
            break
    
    if list_tool:
        try:
            result = await asyncio.to_thread(
                list_tool._run,
                collection_name="documents"
            )
            
            print("\n📋 Embedded Documents:")
            result_data = json.loads(result) if isinstance(result, str) else result
            print(json.dumps(result_data, indent=2))
            
        except Exception as e:
            print(f"❌ Listing failed: {e}")
    
    # Test searching documents
    print("\n4. Testing document search...")
    search_tool = None
    for tool in qdrant_tools:
        if tool.name == "qdrant_memory_search_documents":
            search_tool = tool
            break
    
    if search_tool:
        test_queries = [
            "hypergraph brain",
            "consciousness exploration",
            "semantic relationships"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            try:
                result = await asyncio.to_thread(
                    search_tool._run,
                    query_text=query,
                    limit=3,
                    collection_name="documents"
                )
                
                result_data = json.loads(result) if isinstance(result, str) else result
                if result_data.get("success"):
                    print(f"Found {result_data.get('count', 0)} results:")
                    for i, res in enumerate(result_data.get("results", []), 1):
                        print(f"\n  Result {i}:")
                        print(f"  Score: {res.get('score', 0):.4f}")
                        print(f"  Chunk: {res.get('chunk_index', 0)}")
                        content_preview = res.get('content', '')[:200] + "..." if len(res.get('content', '')) > 200 else res.get('content', '')
                        print(f"  Content: {content_preview}")
                else:
                    print(f"Search failed: {result_data}")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
    
    print("\n✅ Document embedding test complete!")

if __name__ == "__main__":
    asyncio.run(test_document_embedding())
