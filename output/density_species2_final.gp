set terminal png size 800,700 enhanced font 'Helvetica,13'
set output 'output/density_species2_final.png'

set title 'SALR DFT — Species 2 final density'
set xlabel 'x'
set ylabel 'y'
set cblabel 'ρ(x,y)'

set palette rgbformulae 33,13,10    # viridis-like: dark-blue -> yellow
set cbrange [0.197078:0.404044]

set pm3d map interpolate 0,0
set size ratio -1

splot 'output/density_species2_final.dat' using 1:2:3 with pm3d notitle
