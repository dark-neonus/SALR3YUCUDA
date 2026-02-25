set terminal png size 900,500 enhanced font 'Helvetica,13'
set output 'output/convergence.png'

set title 'Picard iteration convergence'
set xlabel 'Iteration'
set ylabel 'L2 error'
set logscale y
set grid
set key off

plot 'output/convergence.dat' using 1:2 with linespoints pt 7 ps 0.4 lw 1.2 lc rgb '#2171b5' notitle
