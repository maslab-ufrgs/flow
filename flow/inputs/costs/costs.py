# script made to generate costs to the grid network

import random


def inBorders(v, h, maxV, maxH):
    if 0 <= h < maxH and 0 <= v < maxV:
        return True
    return False

horizontal = 5
vertical = 5
minCost = 1
maxCost = 50
moves = {(1,0), (-1,0), (0, -1), (0, 1)}
filename = "edgesCost.txt"
bigString = ''

# normal nodes
for v in range(vertical):
    for h in range(horizontal):
        for dV, dH in moves:
            if inBorders(v + dV, h + dH, vertical, horizontal):
                edge = "from{}to{}".format(str(v) + str(h), str(v + dV) + str(h + dH))
                cost = random.randint(minCost, maxCost)
                bigString += edge + ',' + str(cost) + '\n'

# artificial nodes
for v in range(vertical):
    for h in range(horizontal):
        edge = "fromArtificial{}to{}".format(str(v) + str(h), str(v) + str(h))
        cost = 0
        bigString += edge + ',' + str(cost) + '\n'

# write in costs file
with open(filename, 'w') as file:
    file.write('# type: node_to_node, edge_to_node_pd, edge_to_node_list, edges_toll_cost\n')
    file.write('edges_toll_cost\n')
    file.write('# number of horizontal nodes (integer), number of vertical nodes (integer)\n')
    file.write('{}, {}\n'.format(horizontal, vertical))
    file.write('# edge (string), edge cost (float)\n')
    file.write(bigString[:-1])
