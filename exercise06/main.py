'''
Author: Dehui Lin
Title: Exercise 06
'''

from collections import defaultdict, deque
import numpy
import sys

class Graph:
    def __init__(self, beta, negUnary):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
        self.unaries = {}
        self.beta = beta
        self.negUnary = False
     
    def unary(self):
        
        if self.negUnary:
            #(-1, 1)
            energy = numpy.random.random() - numpy.random.random()
        else:
            # [0, 1)
            energy = numpy.random.random()
        
        return energy 
    
    def setPottsBeta(self, beta):
        self.beta = beta
        
    def potts(self, equal_):
        if equal_:
            return 0
        else:
            return self.beta
        
        return 0
       
    def add_node(self, value):
        self.nodes.add(value)
        self.unaries[value] = self.unary()

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance + self.unaries[to_node]
        self.distances[(to_node, from_node)] = distance + self.unaries[to_node]
    
    
    def construct(self, numOfVar):
        
        if (numOfVar < 2):
            print('Error: too few variables')
            sys.exit()
        
        self.add_node('src')
        self.add_node('dest')
        
        # add all nodes first
        for i in range(numOfVar):
            self.add_node(str((i+1)*10+0))
            self.add_node(str((i+1)*10+1))

        # assign valid edges between nodes
        for i in range(numOfVar-1):
            self.add_edge(str((i+1)*10+0), str((i+2)*10+0), self.potts(True))
            self.add_edge(str((i+1)*10+0), str((i+2)*10+1), self.potts(False))
            self.add_edge(str((i+1)*10+1), str((i+2)*10+0), self.potts(False))
            self.add_edge(str((i+1)*10+1), str((i+2)*10+1), self.potts(True))
            
        self.add_edge('src', '10', 0)
        self.add_edge('src', '11', 0)
        
        self.add_edge(str(numOfVar*10+0), 'dest', 0)
        self.add_edge(str(numOfVar*10+1), 'dest', 0)
        
        

def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes: 
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
    
        if min_node is None:
            break
        
#        print(min_node)
#        print('end min node')
        
        nodes.remove(min_node)
        current_weight = visited[min_node]
    
        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path

def printPath(p, endNode):
    
    p_copy = dict(p)
    pp = deque()
    pp.appendleft(endNode)
    
    node = endNode
    
    while True:
        if node in p_copy:
            pp.appendleft(p[node])
            del p_copy[node]
            node = p[node]
        else:
            break
        
    print(pp)

if __name__ == '__main__':
    
    numpy.random.seed(10)
    numOfVariables = 20
    
    
    useNegativeUnary = False

    for beta in [0.01, 0.1, 0.2, 0.5, 1.0]:
        print('Beta: ', beta)
        graph = Graph(beta, useNegativeUnary)
        graph.construct(numOfVariables)
        
        #for i in graph.distances:
            #print(i, ' : ', graph.distances[i])
        
        v, p = dijsktra(graph, 'src')
        print(p)
        printPath(p, 'dest')
        print('\n')
        
    '''
    
    useNegativeUnary = True
    
    for beta in [-1.0, -0.1, -0.01, 0.01, 0.1, 0.2, 0.5, 1.0]:
        print('Beta: ', beta)
        graph = Graph(beta, useNegativeUnary)
        graph.construct(numOfVariables)
        
        #for i in graph.distances:
            #print(i, ' : ', graph.distances[i])
        
        v, p = dijsktra(graph, 'src')
        #print(p)
        printPath(p, 'dest')
        print('\n')
    
    '''
    
    pass