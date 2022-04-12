import numpy as np
graph = [
    [0,1,2,3,4],
    [1,2,3,3,4]
]

embeddings = [
    [1,2,3],
    [2,6,5],
    [2,3,7],
    [7,8,6],
    [1,0,0]
]

w = [1,1,1,1,1]

for i in range(len(graph[0])):
    temp_roots = []

# enumerate()函数用以获取数组的index和values
    for index,value in enumerate(graph[0]):
        if index == i:
            temp_roots.append(graph[0][index])
    temp_roots.append(i)
    around = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    for every_node_id in temp_roots:
        around[every_node_id] = embeddings[every_node_id]
    embeddings[i] = np.matmul(np.array(w),np.array(around))

print(embeddings)
