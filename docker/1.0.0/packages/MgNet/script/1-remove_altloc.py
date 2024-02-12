import os
import sys

inpdb = sys.argv[1]
outpdb = sys.argv[2]

with open(inpdb) as f:
    pdblines = f.readlines()
with open(outpdb,'w') as f:
    for l in pdblines:
        record = l[0:6]
        if (record == "ATOM  " or record == "HETATM") and (l[16] == ' ' or l[16] == 'A'):
            s = list(l)
            #print(l)
            s[16] = ' '
            l = "".join(s)
            #print(l)
            f.write(l)
