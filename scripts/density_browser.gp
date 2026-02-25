#!/usr/bin/env gnuplot
#
# density_browser.gp — Interactive frame-by-frame density browser
#
# Usage (from project root):
#   gnuplot scripts/density_browser.gp
#   gnuplot -e "output_dir='output'" scripts/density_browser.gp
#
# ── Controls ───────────────────────────────────────────────────────────────
#   Right arrow  /  n   next frame
#   Left  arrow  /  p   previous frame
#   Home                jump to first frame
#   End                 jump to last frame
#   q                   quit
#
# The timeline never advances on its own — every step is manual.
# ───────────────────────────────────────────────────────────────────────────

# ── Output directory (override with -e "output_dir='...'" on command line) --
if (!exists("output_dir")) { output_dir = "output" }

# ── Collect file lists ------------------------------------------------------ 
# ls -v = version-sort: handles zero-padded numbers correctly (000100 < 000200)
# Iteration snapshots first (sorted), then the final file if it exists.
_iter1 = system("ls -v ".output_dir."/density_species1_iter_*.dat 2>/dev/null")
_iter2 = system("ls -v ".output_dir."/density_species2_iter_*.dat 2>/dev/null")
_fin1  = system("test -f ".output_dir."/density_species1_final.dat && echo ".output_dir."/density_species1_final.dat")
_fin2  = system("test -f ".output_dir."/density_species2_final.dat && echo ".output_dir."/density_species2_final.dat")

# Concatenate: iter files + final file (strip leading/trailing whitespace)
files1 = (_iter1 eq "" ? "" : _iter1) . (_fin1 eq "" ? "" : " " . _fin1)
files2 = (_iter2 eq "" ? "" : _iter2) . (_fin2 eq "" ? "" : " " . _fin2)

# Trim leading spaces that may appear when _iter is empty
files1 = system("echo '".files1."' | xargs")
files2 = system("echo '".files2."' | xargs")

N = words(files1)
if (N == 0) {
    print "ERROR: No density_species*_iter_*.dat or *_final.dat files found in '".output_dir."/'."
    print "Run the simulation first, then launch this browser."
    exit
}
if (words(files2) != N) {
    print "WARNING: species1 and species2 file counts differ — some frames may mismatch."
}
print sprintf("Loaded %d frames. Use Right/Left arrows (or n/p) to navigate.", N)

# ── Terminal & appearance ───────────────────────────────────────────────────
set terminal qt size 1450,690 enhanced font "Sans,10" title "SALR Density Browser"

set xlabel "x" font ",10"
set ylabel "y" font ",10"
set zlabel "{/Symbol r}" rotate font ",10"
set ticslevel 0         # z-axis starts at the xy-plane base
set key off
set cbrange [*:*]       # auto-scale colour per frame; set fixed range here if desired
set palette defined (0 "#440154", 0.25 "#31688e", 0.5 "#35b779", 0.75 "#fde725", 1 "#ff0000")

# ── Core plotting routine stored as a string and eval'd on every redraw ─────
#
# Using 1:2:3:3  →  x=col1, y=col2, z=col3, colour=col3
# pt 7 = filled circle,  ps 0.5 = point size
# noenhanced on set title prevents _ in filenames from being interpreted as subscript
#
DRAW = \
  "f1 = word(files1, idx); f2 = word(files2, idx);" \
. "label = (strstrt(f1,'final')>0 ? 'FINAL' : sprintf('iter %d', int(real(system(\"echo '\".f1.\"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "set multiplot layout 1,2" \
. "  title sprintf('Frame %d / %d   —   %s   [← p  n →  to navigate]', idx, N, label)" \
. "  font ',12';" \
. "set title '{/Symbol r}_1(x,y)   species 1' font ',11';" \
. "splot f1 u 1:2:3:3 w p pt 7 ps 0.45 lc palette notitle;" \
. "set title '{/Symbol r}_2(x,y)   species 2' font ',11';" \
. "splot f2 u 1:2:3:3 w p pt 7 ps 0.45 lc palette notitle;" \
. "unset multiplot"

# ── Event loop ──────────────────────────────────────────────────────────────
# `pause mouse keypress` blocks here until a key press (or mouse click)
# occurs inside the plot window.  After the pause:
#   MOUSE_CHAR  = string  ("n", "p", "q", …)  for printable keys
#   MOUSE_KEY   = integer for special keys:
#                   65363 = Right arrow
#                   65361 = Left  arrow
#                   65360 = Home
#                   65367 = End
idx = 1
eval DRAW

running = 1
while (running) {
    pause mouse keypress "  [n/→ next   p/← prev   Home   End   q quit]"

    kc = int(MOUSE_KEY)
    ch = MOUSE_CHAR

    # Right arrow or 'n'  →  next frame
    if (kc == 65363 || ch eq "n") {
        idx = (idx < N) ? idx+1 : N
        eval DRAW
    }
    # Left arrow or 'p'  →  previous frame
    if (kc == 65361 || ch eq "p") {
        idx = (idx > 1) ? idx-1 : 1
        eval DRAW
    }
    # Home  →  first frame
    if (kc == 65360) {
        idx = 1
        eval DRAW
    }
    # End  →  last frame
    if (kc == 65367) {
        idx = N
        eval DRAW
    }
    # 'q' or 'Q'  →  quit
    if (ch eq "q" || ch eq "Q") {
        running = 0
    }
}
