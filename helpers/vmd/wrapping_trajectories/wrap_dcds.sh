#! /bin/bash

for i in *.dcd;do

/Applications/VMD/VMD\ 1.9.4a38.app/Contents/MacOS/startup.command -e wrap_dcd.tcl  -args *pdb $i

mv traj.dcd $i

done
