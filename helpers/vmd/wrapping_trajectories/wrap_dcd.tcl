mol new [lindex $argv 0] waitfor all 
foreach filename [lrange $argv 1 $argc-1] {
  mol addfile $filename waitfor all 
}


pbc set {12.42 12.42 12.42} -all

pbc wrap -all

animate write dcd {traj.dcd} beg 1 end -1 skip 1 0 

quit


