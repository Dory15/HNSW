from HSNW import HSNW
from Node import Node
from HSNWGraphics import HNSWGraphics
import numpy as np

def main():
    hsnw = HSNW(dimension=2, M=5)
    
    # Populate the structure with 20 randomized data points
    for i in range(20):
        node = Node(id=i, vector=np.random.rand(2))
        hsnw.add_node(node)

    # Setup execution query vector parameters
    query_vec = np.random.rand(2)
    result_id, result_similarity = hsnw.query(query_vec)
    
    print(f"Closest node ID: {result_id}, Similarity: {result_similarity}")
    print("Query vector:", query_vec)

    # 🚀 CLEAN INVOCATION OF THE SEPARATED GRAPHICS CLASS
    graphics = HNSWGraphics(hsnw=hsnw, query_vec=query_vec, result_id=result_id)
    graphics.draw()

if __name__ == "__main__":
    main()
