import sys
import csv

with open("nrlist_3.50_all.csv") as f:
    nrDict = {}
    csv_reader = csv.reader(f, delimiter=',', quotechar='"')
    for row in csv_reader:
        print(f'\t{row[0]} \t {row[1]} \t {row[2]}')
        Equivalence = row[0].strip()
        for PDB in row[2].split(","):
            PDB = PDB.strip()
            PDBid = PDB[0:4]
            if PDBid in nrDict:
                if PDB in nrDict[PDBid]:
                    nrDict[PDBid][PDB].add(Equivalence)
                else:
                    s1 = set()
                    s1.add(Equivalence)
                    nrDict[PDBid][PDB] = s1
            else:
                s2 = set()
                s2.add(Equivalence)
                d = {}
                d[PDB] = s2
                nrDict[PDBid] = d
    print(nrDict)
    
with open("./ion_statistics.csv") as f:
    line_count = 0
    with open("./ion_statistics_update.csv","w") as fw:
        for row in f:
            if line_count == 0:
                fw.write(row)
            else:
                if (int(row.split(",")[4]) != 0) and (int(row.split(",")[2]) >= 10):
                    fw.write(row)
            line_count += 1
        
        

nr_statistics = {"not_included":set()}
with open("./ion_statistics_update.csv") as f:
    csv_reader = csv.DictReader(f, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')  
        PDB = row["PDB"].upper()
        outline = ""
        for key,value in row.items():
            outline += str(value)+","
        
        if PDB in nrDict:
            for key,value in nrDict[PDB].items():
                value = str(value)
                if value in nr_statistics:
                    nr_statistics[value].add(outline+str(key))
                else:
                    o = set()
                    o.add(outline+str(key))
                    nr_statistics[value] = o
        else:
            nr_statistics["not_included"].add(outline)
        line_count += 1
    print(f'Processed {line_count} lines.')
    print(nr_statistics)
    
    
    

with open("nr_statistics.csv","w") as f:
    for key,value in nr_statistics.items():
        f.write(str(key)+"\n")
        for v in value:
            f.write(v+"\n")
