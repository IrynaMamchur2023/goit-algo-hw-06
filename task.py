import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

G = nx.Graph()

G.add_node("A", pos=(0, 0))
G.add_node("B", pos=(1, 1))
G.add_node("C", pos=(1, -1))
G.add_node("D", pos=(2, 0))

G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("C", "D")
G.add_edge("D", "A")

pos = nx.get_node_attributes(G, "pos")

nx.draw_networkx(G, with_labels=True, node_color='lightblue', node_size=1500, font_size=12, font_weight='bold')
plt.title("Мережа доріг у місті")
plt.show()

print("Кількість вершин у графі:", G.number_of_nodes())
print("Кількість ребер у графі:", G.number_of_edges())
print("Ступінь кожної вершини у графі:")
for node in G.nodes():
    print(f"{node}: {G.degree[node]}")  

def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next_node in graph[vertex]:
            if next_node not in path:
                if next_node == goal:
                    yield path + [next_node]
                else:
                    stack.append((next_node, path + [next_node]))

def bfs_paths(graph, start, goal):
    queue = deque([(start, [start])])
    while queue:
        (vertex, path) = queue.popleft()
        for next_node in graph[vertex]:
            if next_node not in path:
                if next_node == goal:
                    yield path + [next_node]
                else:
                    queue.append((next_node, path + [next_node]))

start_node = "A"
goal_node = "D"
dfs_result = list(dfs_paths(G, start_node, goal_node))

bfs_result = list(bfs_paths(G, start_node, goal_node))

print("Шляхи, знайдені за допомогою DFS:")
for path in dfs_result:
    print(path)
print("\nШляхи, знайдені за допомогою BFS:")
for path in bfs_result:
    print(path)

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor in graph[current_vertex]:
            weight = graph[current_vertex][neighbor].get("weight", 1) 
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

G_with_weights = G.copy()
for u, v in G_with_weights.edges():
    G_with_weights.edges[u, v]["weight"] = 1 

all_shortest_paths = {}
for node in G_with_weights.nodes():
    all_shortest_paths[node] = dijkstra(G_with_weights, node)

for node, shortest_paths in all_shortest_paths.items():
    print(f"Найкоротші шляхи від вершини {node}:")
    for target, shortest_distance in shortest_paths.items():
        print(f"Вершина {target}: Довжина шляху - {shortest_distance}")