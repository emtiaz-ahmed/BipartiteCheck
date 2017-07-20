import numpy
import math
import itertools
import time
import sys
from mpi4py import MPI
from random import randint
from itertools import islice

machineSize = 0
numOfVertex = 0
chunkSize   = 0
adjacency   = {}
vertexColor = {}
sourceNode  = 1
pVertex     = {}
d           = []

# This function load the input graph from file and distribute the vertices into different machine 
def load_graph(rank, machineSize, comm):
    words = []
    while True:
        if rank == 0:                                                    # rank 0 is the master machine, and task of master machine starts here
            f = open("input.txt","r")                                    # input file
            
            for line in f.readlines():
                words = line.split()
                p = (int(words[0])-1) % machineSize                      # hashing vertices according to their ID
                if machineSize == 1:
                    if not int(words[0]) in adjacency:
                        adjacency[int(words[0])] = [int(words[1])]
                    else:
                        adjacency[int(words[0])].append(int(words[1]))
                else:
                    if p == 0:
                        if not int(words[0]) in adjacency:
                            adjacency[int(words[0])] = [int(words[1])]
                        else:
                            adjacency[int(words[0])].append(int(words[1]))
                    else:
                        comm.send(words[0], dest=p, tag = 1)              # send vertices to respective machine
                        comm.send(words[1], dest=p, tag = 2)              
            if machineSize > 1:
                for i in range(1,machineSize):
                    comm.send("done", dest=i,tag = 1)
                    comm.send("done", dest=i,tag = 2)
            break

        else:                                                             # task of other machines starts here
            buf1 = comm.recv(source = 0, tag = 1)                         # recieve the message
            buf2 = comm.recv(source = 0, tag = 2)
            if buf1=="done":
                break
            else:
                if not int(buf1) in adjacency:
                    adjacency[int(buf1)] = [int(buf2)]
                else:
                    adjacency[int(buf1)].append(int(buf2))
    

def find_source(rank,comm):                                               # select source as the 1st vertex in master machine
    if rank == 0:
        for k,v in adjacency.items():
            sourceNode = k
            break
    else:
        sourceNode = None
    sourceNode = comm.bcast(sourceNode, root=0)                           # broadcast the source vertex ID to everyone

def coloring_default():                                                   # initially assign default negative color to every node
    for k in adjacency.keys():
        vertexColor[k] = 0
    

def parent_default():                                                     # initially assign default parents for every vertex
    for k in adjacency.keys():
        pVertex[k] = 0
    


# This function is the main function of our algorithm, it checks the given graph is bipartite or not
def bipartite_check(rank,comm, machineSize):                              
    global d
    flag = 0
    flagBreak = 0
    for k in adjacency.keys():
        for v in adjacency[k]:
            if not v in adjacency.keys():
                flagBreak = flagBreak + 1
   

    if rank == 0:                                                         # task of master machine starts here
        currentNodes = [sourceNode]        
        nextNodes = []
        vertexColor[sourceNode] = 1
        parent  = -1       

        while True:
            returnVal = common_work(currentNodes, adjacency, vertexColor, nextNodes, parent, machineSize, comm)
            if returnVal == 1:
                return 1
            if machineSize == 1:
                break
            
            if flag == flagBreak:
                break

            data = comm.recv(source = MPI.ANY_SOURCE, tag = 5)            # receive message from other machines
            uColor = data[0]
            currentNodes = [data[1]]
            parent = data[2]                                              
            flag = flag + 1            
            
            if vertexColor[data[1]] == 0:
                vertexColor[data[1]] = 3 - uColor                         # assign color of current node if it is not colored already
                pVertex[data[1]] = parent                                 # assign parent of current vertex
            elif vertexColor[data[1]] == uColor:                          # if parent and current node has same color then return not bipartite
                d.append( data[1])                                        # return conflicted edge
                d.append( parent)                
                return 1
                break
            else:
                currentNodes = []
            nextNodes = []
            
    else:                                                                 # task of other machines starts here
        
        while True:
            data = comm.recv(source = MPI.ANY_SOURCE, tag = 5)            # receive messages from other machines
            uColor = data[0]
            currentNodes = [data[1]]
            parent = data[2]
            flag = flag + 1
            
            if vertexColor[data[1]] == 0:                                 # assign color to current node if not colored
                vertexColor[data[1]] = 3 - uColor
                pVertex[data[1]] = parent                                 # assign parent of current vertex
            elif vertexColor[data[1]] == uColor:                          # if parent and current vertex has same color then return not bipartite
                d.append(data[1])                                         # return conflicted edge
                d.append(parent)               
                return 1
                break
            else:
                currentNodes = []            
            nextNodes = []          
           
            common_work(currentNodes, adjacency, vertexColor, nextNodes, parent, machineSize, comm)            
            if flag == flagBreak:
                break
       
    return 0

# This function is do the common tasks of all machines
# This function sends the message to respective machines 
def common_work(currentNodes, adjacency, vertexColor, nextNodes, parent, machineSize, comm):
    global d
    while True:
        if len(currentNodes) > 0:
            for u in currentNodes:
                for v in adjacency[u]:
                    if v in adjacency.keys():
                        if vertexColor[v] == 0:                           # assign color to the current node if not colored
                            vertexColor[v] = 3 - vertexColor[u]
                            pVertex[v] = u
                            nextNodes.append(v)
                        elif vertexColor[v] == vertexColor[u]:            # if parent and current vertex has same color then return not bipartite                           
                            d.append(v)
                            d.append(u)                            
                            return 1
                            break
                    else:
                        buf = []
                        buf.append(vertexColor[u])
                        buf.append(v)
                        buf.append(u)                        
                        comm.send(buf, dest = (v-1) % machineSize, tag = 5)   # send message to respective machine
            currentNodes = nextNodes
            nextNodes = []
        else:
            break
    return 0

# This function check the odd cycle if the graph is not bipartite
# print the edges whose makes the odd cycle
def check_oddCycle(d,pVertex):
    u = d[0]
    v = d[1]
    print("This graph is not bipartite because Below edges creates oddCycle:")
    print(u,v)
    while u != v:
        print(u,pVertex[u])
        print(v,pVertex[v])
        u = pVertex[u]
        v = pVertex[v]

# This is the main function
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()                                                 # find the rank of current machine                      
    machineSize = comm.Get_size()                                          # find the number of machines

    load_graph(rank, machineSize,comm)                                     # load the graph from input    
    find_source(rank,comm)                                                 # find the source    
    coloring_default()                                                     # assign default color to every vertex
    parent_default()                                                       # initialize default parents

    comm.Barrier()                                                         # wait until all machine are done with previous tasks
    startTime = MPI.Wtime()    
    localResult = bipartite_check(rank,comm,machineSize)                   # save the local return values
    globalResult = 5                                                    
    globalResult = comm.reduce(localResult, op = MPI.SUM, root = 0)        # collect all local data from different machine and save into master
    if rank == 0:
        if globalResult == 0:
            print("This is a Bipartite graph")
        else:
            print("This is not a Bipartite graph")
    comm.Barrier()
    endTime = MPI.Wtime()

    if machineSize == 1 and globalResult != 0:                             # if not bipartite then check the odd cycle
        check_oddCycle(d,pVertex)     

    if rank == 0:
        print("Total running time: ",endTime-startTime)                    # return running time of the algorithm


if __name__ == '__main__':
    main()

