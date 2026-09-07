import numpy as np

class HSNW:
    
    def __init__(self, dimension, M):
        """
        M is connections per node.
        mL is a probability factor used to promote a node to the next level.
        """
        self.M = M
        self.dimension = dimension
        self.mL = 1 / np.log(M)
        self.entry_node = None
        self.max_level = -1
        self.nodes = {}
    
    def get_cosine_similarity(self, v1, v2):
        """
        We don't need the magnitude multiplication because we are using normalized vectors.
        """
        return float(np.dot(v1, v2))
        
    def _get_max_random_level(self):
        """
        Assigns randomly the maximum level that a node can be part of.
        """
        return int(-np.log(np.random.uniform(0, 1)) * self.mL)
    
    def search_in_layer(self, query_vec, start_node_id, level):
        """
        Performs a greedy local-optimum search for the closest node inside a single level.
        """
        start_node = self.nodes[start_node_id]
        current_similarity = self.get_cosine_similarity(query_vec, start_node.vector)
        
        while True:
            neighbors = start_node.neighboards_per_level.get(level, [])
            best_candidate = start_node
            best_similarity = current_similarity

            # Explore all neighbors at the current level to find the best candidate
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                similarity = self.get_cosine_similarity(query_vec, neighbor.vector)
                if similarity > best_similarity:
                    best_candidate = neighbor
                    best_similarity = similarity

            # If no better candidate is found among neighbors, the search will terminate
            if best_candidate == start_node:
                break
                
            start_node = best_candidate
            current_similarity = best_similarity
            
        return start_node.id, current_similarity

    def add_node(self, node):
        self.nodes[node.id] = node
        node_level = self._get_max_random_level()
        
        # Initialize levels for the new node
        for level in range(node_level + 1):
            if level not in node.neighboards_per_level:
                node.neighboards_per_level[level] = []
                
        # If the HSNW is empty, initialize the entry node and max level
        if self.entry_node is None:
            self.entry_node = node
            self.max_level = node_level
            return

        current_node = self.entry_node
        
        # 1. Top-down fast routing
        for level in range(self.max_level, node_level, -1):
            # Search for the closest node at the current level before descending further
            current_node_id, _ = self.search_in_layer(node.vector, current_node.id, level)
            current_node = self.nodes[current_node_id]
            
        # 2. Connection Phase (from the overlap down to level 0)
        for level in range(min(node_level, self.max_level), -1, -1):
            # Search for the closest node at the current level before attempting connections
            current_node_id, _ = self.search_in_layer(node.vector, current_node.id, level)
            current_node = self.nodes[current_node_id]

            # Retrieve the current node's neighbors at this level
            neighbors = current_node.neighboards_per_level.get(level, [])
            
            if len(neighbors) < self.M:
                # Room available: Connect bidirectionally
                if node.id not in neighbors:
                    neighbors.append(node.id)
                current_node.neighboards_per_level[level] = neighbors
                if current_node.id not in node.neighboards_per_level[level]:
                    node.neighboards_per_level[level].append(current_node.id)
            else:
                # Evaluate existing neighbors
                neighbor_scores = []
                for n_id in neighbors:
                    n_node = self.nodes[n_id]
                    sim = self.get_cosine_similarity(current_node.vector, n_node.vector)
                    neighbor_scores.append((sim, n_id))

                # Sort neighbors by similarity to identify the worst one for potential pruning
                neighbor_scores.sort(key=lambda x: x[0])
                worst_similarity, worst_neighbor_id = neighbor_scores[0]
                new_node_similarity = self.get_cosine_similarity(current_node.vector, node.vector)
                
                # ✅ CONNECTIVITY GUARD: Only prune if the target node won't be left isolated (has > 1 neighbor)
                worst_neighbor_node = self.nodes[worst_neighbor_id]
                worst_neighbor_links = worst_neighbor_node.neighboards_per_level.get(level, [])

                # Determine if it's safe to replace the worst neighbor with the new node based on similarity and connectivity
                if new_node_similarity > worst_similarity and len(worst_neighbor_links) > 1:
                    neighbors.remove(worst_neighbor_id)
                    if current_node.id in worst_neighbor_links:
                        worst_neighbor_links.remove(current_node.id)
                    
                    neighbors.append(node.id)
                    current_node.neighboards_per_level[level] = neighbors
                    if current_node.id not in node.neighboards_per_level[level]:
                        node.neighboards_per_level[level].append(current_node.id)
                else:
                    # Fallback: If pruning is unsafe, append to the new node directly to preserve graph integrity
                    if current_node.id not in node.neighboards_per_level[level]:
                        node.neighboards_per_level[level].append(current_node.id)

            # ✅ CRITICAL FALLBACK FOR HIGHER LAYERS:
            # If the new node didn't manage to establish any connections at this level, 
            # force a fallback bridge connection to the local cluster head to avoid an isolated layer island.
            if len(node.neighboards_per_level[level]) == 0:
                node.neighboards_per_level[level].append(current_node.id)
                if node.id not in current_node.neighboards_per_level[level]:
                    current_node.neighboards_per_level[level].append(node.id)

        # 3. Global update if the node sets a new record height
        if node_level > self.max_level:
            self.entry_node = node
            self.max_level = node_level

    def query(self, query_vec):
        if self.entry_node is None:
            return None, 0.0
            
        current_node_id, current_similarity = self.search_in_layer(query_vec, self.entry_node.id, self.max_level)
        for level in range(self.max_level - 1, -1, -1):
            current_node_id, current_similarity = self.search_in_layer(query_vec, current_node_id, level)
            
        return current_node_id, current_similarity
