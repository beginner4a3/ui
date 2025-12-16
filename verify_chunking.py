
import re
import numpy as np

def test_chunking():
    text = "Hello world! This is a test. How are you? I am fine."
    chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = [c.strip() for c in chunks if c.strip()]
    
    print(f"Original: {text}")
    print(f"Chunks: {chunks}")
    
    expected = ["Hello world!", "This is a test.", "How are you?", "I am fine."]
    if chunks == expected:
        print("✅ Chunking works!")
    else:
        print("❌ Chunking failed!")
        
    # Test concatenation
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    combined = np.concatenate([arr1, arr2])
    print(f"Combined array: {combined}")
    
    if len(combined) == 6:
        print("✅ Concatenation works!")
    else:
        print("❌ Concatenation failed!")

if __name__ == "__main__":
    test_chunking()
