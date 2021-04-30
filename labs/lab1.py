#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the maxRegion function below.
def maxRegion(grid):
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    def fill(i, j):
        viz[i][j] = 1
        ct = 1
        Max = ct
        for d in range(8):
            if i+dx[d] >= 0 and j+dy[d] >= 0 and i+dx[d] < n and j+dy[d] < m:
                if grid[i+dx[d]][j+dy[d]] == 1 and viz[i+dx[d]][j+dy[d]] == 0: 
                    ct += fill(i+dx[d], j+dy[d])
        return ct
    
    viz = [[0 for _ in range(m)] for _ in range(n)]
    MaxG = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and viz[i][j] == 0:
                MaxG = max(MaxG, fill(i, j))
    return MaxG    
            

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    m = int(input())

    grid = []

    for _ in range(n):
        grid.append(list(map(int, input().rstrip().split())))

    res = maxRegion(grid)

    fptr.write(str(res) + '\n')

    fptr.close()

"""
def graphReader(pathFile):
    fileContent = open(pathFile, "r")
    lines = []

    for line in fileContent:
        lines.append(line.replace('\n', ""))

    numberNodes = int(lines[0])

    addiacent_list = []
    for i in range(0, numberNodes):
        addiacent_list.append([])

    for i in range(1, len(lines)):
        for node in lines[i].split(' '):
            if node != '':
                addiacent_list[i - 1].append(int(node))

    return numberNodes, addiacent_list

def parcugere_adancime(numberNodes, addList, startNode, scopeNode):
	viz = [0] * numberNodes
	father = [0] * numberNodes
	def dfs(nod, tata):
		father[nod] = tata
		viz[nod] = 1
		am_atins = scopeNode == nod
		for vecin in addList[nod]:
			if dfs(vecin, nod):
				am_atins = True
		return am_atins
	
	dfs

def bfs(numberNodes, addList, startNode, scopeNode):
  viz = [False for _ in range(numberNodes)]
  father = [-1 for _ in range(numberNodes)]

  q = [startNode]
  viz[startNode] = True

  while len(q) != 0:
    nod = q[0]
    print("Here")
    q.pop(0)
    for vec in addList[nod]:
      if not viz[vec]:
        q.append(vec)
        viz[vec] = True
        father[vec] = nod
        if vec == scopeNode:
          q = []
          break
    
  if not viz[scopeNode]:
    return []

  roadToScope = []
  while scopeNode != -1:
    roadToScope.append(scopeNode)
    scopeNode = father[scopeNode]

  roadToScope = roadToScope[::-1]
  return roadToScope

if __name__ == '__main__':
    numberNodes, addList = graphReader("graphStructure.txt")
    roadToScope = bfs(numberNodes, addList, 0, 7)
    print(roadToScope)
"""