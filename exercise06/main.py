from collections import defaultdict

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
        
    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance


def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes: 
        print(visited)
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
    
        if min_node is None:
            break
        
        print(min_node)
        print('end min node')
        
        nodes.remove(min_node)
        current_weight = visited[min_node]
    
        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path


if __name__ == '__main__':
    
    graph = Graph()
    
    graph.add_node('n00')
    graph.add_node('n10')
    graph.add_node('n11')
    graph.add_node('n20')
    graph.add_node('n21')
    graph.add_node('n30')
    
    graph.add_edge('n00', 'n10', 1000)
    graph.add_edge('n00', 'n11', 1000)
    graph.add_edge('n10', 'n20', 4)
    graph.add_edge('n10', 'n21', 1)
    graph.add_edge('n11', 'n20', -2)
    graph.add_edge('n11', 'n21', 3)
    graph.add_edge('n20', 'n30', 1000)
    graph.add_edge('n21', 'n30', 1000)
    
    print(graph.nodes)
    print(graph.edges)
    print(graph.distances)
    
    v, p = dijsktra(graph, 'n00')
    
    print(v)
    print(p)
    
    
    pass