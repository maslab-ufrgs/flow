import random
import csv
header = ["edge_id", "cost"]
lanes = ["A1A","AB","AC","B1B","AD","BA","BD", "BE","CA","CD","CF","CG","DA","DB","DC","DE","DG","DH","EB","ED","EH", "FC","FG","FI","GC","GD","GF","GH","GJ","GK","HD","HE","HG","HK","IF","IJ","IL","JG","JI", "JK","JL","JM","KG","KH","KJ","KM","LI","LJ","LL1","MJ","MK","MM1"]
spawnNodes = ["A1A", "B1B", "MM1", "LL1"]
lbCost = 1000
ubCost = 5000
filename = "genCosts.csv"
with open(filename, "w") as file:
    csvWriter = csv.writer(file)
    csvWriter.writerow(header)
    for lane in lanes:
        if lane not in spawnNodes:
            cost = random.randint(lbCost, ubCost)
        else:
            cost = 0
        csvWriter.writerow([lane, cost])