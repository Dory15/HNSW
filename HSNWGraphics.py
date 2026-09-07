import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HNSWGraphics:
    def __init__(self, hsnw, query_vec=None, result_id=None):
        self.hsnw = hsnw
        self.query_vec = query_vec
        self.result_id = result_id
        
        # Dynamically calculate the maximum level present in the graph
        self.max_level = max([max(n.neighboards_per_level.keys()) for n in hsnw.nodes.values() if n.neighboards_per_level]) if hsnw.nodes else 0
        
        # FIX: Using the modern Matplotlib API to generate a dynamic color palette
        base_cmap = plt.colormaps['plasma']
        
        # Scale indices between 0.0 and 1.0 to sample cleanly from the color map
        self.layer_colors = {}
        for lvl in range(self.max_level + 1):
            factor = lvl / self.max_level if self.max_level > 0 else 0.0
            self.layer_colors[lvl] = base_cmap(factor)
            
        self.layer_styles = {0: ':', 1: '--'} # Levels 2 and above will default to a solid line '-'

    def draw(self):
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. PLOT THE QUERY VECTOR (If provided)
        if self.query_vec is not None:
            ax.plot([self.query_vec[0], self.query_vec[0]], 
                    [self.query_vec[1], self.query_vec[1]], 
                    [0, self.max_level], 
                    color='green', linestyle='-.', alpha=0.4, linewidth=1.5)
            
            for lvl in range(self.max_level + 1):
                ax.scatter(self.query_vec[0], self.query_vec[1], lvl, color='green', s=130, edgecolors='black', marker='^', zorder=5)
                if lvl == self.max_level:
                    ax.text(self.query_vec[0], self.query_vec[1], lvl + 0.08, "QUERY 🎯", color='green', horizontalalignment='center', fontsize=9, fontweight='bold')

        # 2. DRAW CONNECTIONS BETWEEN NEIGHBORS (EDGES)
        drawn_lines = set()
        for node_id, node in self.hsnw.nodes.items():
            for level, neighbors in node.neighboards_per_level.items():
                for neighbor_id in neighbors:
                    line_id = tuple(sorted((node_id, neighbor_id))) + (level,)
                    if line_id not in drawn_lines:
                        drawn_lines.add(line_id)
                        
                        neighbor = self.hsnw.nodes[neighbor_id]
                        x_line = [node.vector[0], neighbor.vector[0]]
                        y_line = [node.vector[1], neighbor.vector[1]]
                        z_line = [level, level]

                        ax.plot(x_line, y_line, z_line, 
                                color=self.layer_colors.get(level, 'black'), 
                                linestyle=self.layer_styles.get(level, '-'), 
                                alpha=0.6, linewidth=1.2, zorder=1)

        # 3. DRAW NODES AT EACH LEVEL THEY EXIST IN
        for node_id, node in self.hsnw.nodes.items():
            for level in node.neighboards_per_level.keys():
                x, y, z = node.vector[0], node.vector[1], level
                
                if node_id == self.result_id:
                    point_color = 'cyan'
                    marker_size = 160
                    label_text = f"⭐ RESULT (ID: {node.id})"
                    font_color = 'darkblue'
                else:
                    point_color = self.layer_colors.get(level, 'black')
                    marker_size = 90
                    label_text = f"ID: {node.id}"
                    font_color = 'black'

                ax.scatter(x, y, z, color=point_color, s=marker_size, edgecolors='black', zorder=3)
                ax.text(x, y, z + 0.08, label_text, color=font_color, horizontalalignment='center', fontsize=8, fontweight='bold')

        # 4. AXIS AND CANVAS FINAL LAYOUT CONFIGURATION
        ax.set_title("Hierarchical HNSW Graph Structure in 3D (R2 Vectors + Z Axis Layer Level)", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Vector Dimension X")
        ax.set_ylabel("Vector Dimension Y")
        ax.set_zlabel("Graph Level (HNSW Tree Height)")
        ax.set_zticks(range(self.max_level + 1))
        
        ax.grid(True, linestyle='--', alpha=0.5)

        for lvl in range(self.max_level + 1):
            ax.plot([], [], [], color=self.layer_colors[lvl], linestyle=self.layer_styles.get(lvl, '-'), label=f'Layer {lvl}')
        if self.query_vec is not None:
            ax.plot([], [], [], color='green', marker='^', linestyle='-.', label='Query Vector')
        
        ax.legend(loc='upper left', framealpha=0.9)
        plt.show()
