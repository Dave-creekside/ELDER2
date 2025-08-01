#!/usr/bin/env python3
"""
Simple test of document embedding
"""

import asyncio
import json
import os

# Set environment variables
os.environ['SENTENCE_TRANSFORMER_MODEL'] = 'all-MiniLM-L6-v2'
os.environ['QDRANT_HOST'] = 'localhost'
os.environ['QDRANT_PORT'] = '6333'

async def test_simple_embed():
    print("🧪 Simple Document Embedding Test")
    print("=" * 50)
    
    # Test 1: Direct MCP server call
    print("\n1. Testing direct MCP server connection...")
    
    import subprocess
    import tempfile
    
    # Create test script
    test_script = f'''
import asyncio
import json
import sys
import os

# Add the project directory to the path
sys.path.insert(0, r"{os.getcwd()}")

from mcp_servers.qdrant_memory.server import QdrantMemoryServer

async def test():
    server = QdrantMemoryServer()
    await server.connect_qdrant()
    
    # Test embedding
    result = await server.embed_document(
        file_path="dream_2025-07-28_21-41-28.md",
        chunk_size=500,
        chunk_overlap=100,
        metadata={{"type": "dream_journal"}},
        collection_name="documents"
    )
    
    print(json.dumps(result, indent=2))
    
    await server.close()

asyncio.run(test())
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Run the test script
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(f"Return code: {result.returncode}")
        print(f"\nStdout:\n{result.stdout}")
        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")
            
    finally:
        os.unlink(script_path)

if __name__ == "__main__":
    asyncio.run(test_simple_embed())
