#!/usr/bin/env gnuplot
#
# density_browser_merged.gp — Interactive merged-species density browser
#
# Usage (from project root):
#   gnuplot scripts/density_browser_merged.gp
#   gnuplot -e "output_dir='output'" scripts/density_browser_merged.gp
#
# ── Controls ──────────────────────────────────────────────────────────────
#   n / Right arrow       +1 frame
#   p / Left  arrow       -1 frame
#   ] / Page Down         +10 frames
#   [ / Page Up           -10 frames
#   N (Shift-n) / }       +100 frames
#   P (Shift-p) / {       -100 frames
#   f / Home              first frame
#   l / End               last frame
#   0-9 then Enter        type frame number and press Enter to jump
#   c                     toggle colour clipping (bulk scale <-> full auto)
#   i                     toggle parameter info display (on <-> off)
#   d                     toggle detailed A/alpha coefficients (on <-> off)
#   q                     quit
#   R                     rotate mode (click-drag to rotate, any key to return)
#
# LAYOUT
#   Left:  3D scatter visualization (purple=SALR/rho1, green=Solvent/rho2)
#   Right: 2D heatmap overlay (purple for species 1, green for species 2,
#          transparency proportional to density, black background)
#
# COLOUR SCHEME
#   3D scatter: Purple (#8B008B) for species 1, Green (#2E8B57) for species 2
#   Heatmap:    Purple gradient for species 1, green gradient for species 2
#               Transparency (alpha) based on local density
#               Black background for zero-density regions
# ──────────────────────────────────────────────────────────────────────────

if (!exists("output_dir")) { output_dir = "output" }

# ── Load simulation parameters ─────────────────────────────────────────────
param_file = output_dir."/parameters.cfg"
param_exists = int(system("test -f '".param_file."' && echo 1 || echo 0"))

if (param_exists) {
    # Read grid parameters
    grid_Lx = real(system("grep '^Lx' '".param_file."' | awk -F'=' '{print $2}'"))
    grid_Ly = real(system("grep '^Ly' '".param_file."' | awk -F'=' '{print $2}'"))
    grid_dx = real(system("grep '^dx' '".param_file."' | awk -F'=' '{print $2}'"))
    grid_dy = real(system("grep '^dy' '".param_file."' | awk -F'=' '{print $2}'"))
    grid_nx = int(real(system("grep '^nx' '".param_file."' | awk -F'=' '{print $2}'")))
    grid_ny = int(real(system("grep '^ny' '".param_file."' | awk -F'=' '{print $2}'")))
    grid_bc = system("grep '^boundary_mode' '".param_file."' | awk -F'=' '{gsub(/^[ \\t]+|[ \\t]+$/,\"\",$2); print $2}'")
    
    # Read physics parameters
    phys_T = real(system("grep '^temperature' '".param_file."' | awk -F'=' '{print $2}'"))
    phys_rho1 = real(system("grep '^rho1' '".param_file."' | awk -F'=' '{print $2}'"))
    phys_rho2 = real(system("grep '^rho2' '".param_file."' | awk -F'=' '{print $2}'"))
    phys_rc = real(system("grep '^cutoff_radius' '".param_file."' | awk -F'=' '{print $2}'"))
    
    # Read solver parameters
    solv_xi1 = real(system("grep '^xi1' '".param_file."' | awk -F'=' '{print $2}'"))
    solv_xi2 = real(system("grep '^xi2' '".param_file."' | awk -F'=' '{print $2}'"))
    solv_tol = real(system("grep '^tolerance' '".param_file."' | awk -F'=' '{print $2}'"))
    solv_max = int(real(system("grep '^max_iterations' '".param_file."' | awk -F'=' '{print $2}'")))
    
    # Read interaction parameters (for expandable section)
    A11_1 = real(system("grep '^A_11_1' '".param_file."' | awk -F'=' '{print $2}'"))
    A11_2 = real(system("grep '^A_11_2' '".param_file."' | awk -F'=' '{print $2}'"))
    A11_3 = real(system("grep '^A_11_3' '".param_file."' | awk -F'=' '{print $2}'"))
    a11_1 = real(system("grep '^a_11_1' '".param_file."' | awk -F'=' '{print $2}'"))
    a11_2 = real(system("grep '^a_11_2' '".param_file."' | awk -F'=' '{print $2}'"))
    a11_3 = real(system("grep '^a_11_3' '".param_file."' | awk -F'=' '{print $2}'"))
    
    A12_1 = real(system("grep '^A_12_1' '".param_file."' | awk -F'=' '{print $2}'"))
    A12_2 = real(system("grep '^A_12_2' '".param_file."' | awk -F'=' '{print $2}'"))
    A12_3 = real(system("grep '^A_12_3' '".param_file."' | awk -F'=' '{print $2}'"))
    a12_1 = real(system("grep '^a_12_1' '".param_file."' | awk -F'=' '{print $2}'"))
    a12_2 = real(system("grep '^a_12_2' '".param_file."' | awk -F'=' '{print $2}'"))
    a12_3 = real(system("grep '^a_12_3' '".param_file."' | awk -F'=' '{print $2}'"))
    
    A22_1 = real(system("grep '^A_22_1' '".param_file."' | awk -F'=' '{print $2}'"))
    A22_2 = real(system("grep '^A_22_2' '".param_file."' | awk -F'=' '{print $2}'"))
    A22_3 = real(system("grep '^A_22_3' '".param_file."' | awk -F'=' '{print $2}'"))
    a22_1 = real(system("grep '^a_22_1' '".param_file."' | awk -F'=' '{print $2}'"))
    a22_2 = real(system("grep '^a_22_2' '".param_file."' | awk -F'=' '{print $2}'"))
    a22_3 = real(system("grep '^a_22_3' '".param_file."' | awk -F'=' '{print $2}'"))
    
    print "Parameters loaded from ".param_file
} else {
    print "WARNING: parameters.cfg not found in ".output_dir
    print "Parameter display will be disabled."
}

# ── Collect file lists ─────────────────────────────────────────────────────
_iter1 = system("ls -v ".output_dir."/data/density_species1_iter_*.dat 2>/dev/null")
_iter2 = system("ls -v ".output_dir."/data/density_species2_iter_*.dat 2>/dev/null")
_fin1  = system("test -f ".output_dir."/density_species1_final.dat && echo ".output_dir."/density_species1_final.dat || true")
_fin2  = system("test -f ".output_dir."/density_species2_final.dat && echo ".output_dir."/density_species2_final.dat || true")

files1 = system("echo '".(_iter1).(_fin1 eq "" ? "" : " "._fin1)."' | xargs")
files2 = system("echo '".(_iter2).(_fin2 eq "" ? "" : " "._fin2)."' | xargs")

N = words(files1)
if (N == 0) {
    print "ERROR: no density files found in '".output_dir."/' or '".output_dir."/data/'."
    print "Run the simulation first, then relaunch this browser."
    exit
}
print sprintf("Loaded %d frames.", N)

# ── Compute bulk means from first snapshot (for colour clipping) ───────────
CB_SCALE  = 3.0
rho1_mean = real(system("awk 'NR>1{s+=$3;n++}END{printf \"%.6f\",s/n}' ".word(files1,1)))
rho2_mean = real(system("awk 'NR>1{s+=$3;n++}END{printf \"%.6f\",s/n}' ".word(files2,1)))
cb1_clip   = CB_SCALE * rho1_mean
cb2_clip   = CB_SCALE * rho2_mean
cb_mix_clip = CB_SCALE * (rho1_mean + rho2_mean)
print sprintf("  rho1 mean=%.4f  colour clip=[0,%.3f]", rho1_mean, cb1_clip)
print sprintf("  rho2 mean=%.4f  colour clip=[0,%.3f]", rho2_mean, cb2_clip)
print sprintf("  rho_mix mean=%.4f  colour clip=[0,%.3f]", rho1_mean+rho2_mean, cb_mix_clip)

# Helper functions: clamp density to [0,1] relative to clip value
t1(v) = ( v > 0 ? (v/cb1_clip < 1.0 ? v/cb1_clip : 1.0) : 0.0 )
t2(v) = ( v > 0 ? (v/cb2_clip < 1.0 ? v/cb2_clip : 1.0) : 0.0 )
# Combined RGB as single integer: Purple for species1, Green for species2
hmap_r(a,b) = int(139*t1(a) + 46*t2(b))
hmap_g(a,b) = int(139*t2(b))
hmap_b(a,b) = int(139*t1(a) + 87*t2(b))

clip_mode = 1      # 1=clipped to 3*mean, 0=auto full range
show_params = 1    # 1=show basic params, 0=hide

# ── Terminal & appearance ──────────────────────────────────────────────────
# wxt (Cairo) and qt support per-element font overrides; x11 does not
if (strstrt(GPVAL_TERMINALS, "wxt")) {
    set terminal wxt size 2400,1100 enhanced font "Sans:Bold,14" title "SALR Merged Density Browser"
} else {
    if (strstrt(GPVAL_TERMINALS, "qt")) {
        set terminal qt size 2400,1100 enhanced font "Sans:Bold,14" title "SALR Merged Density Browser"
    } else {
        if (strstrt(GPVAL_TERMINALS, "x11")) {
            # NOTE: x11 ignores per-element font overrides — font sizes in labels/titles
            # will all follow the terminal-level font above.
            set terminal x11 size 2400,1100 enhanced font "Sans:Bold,14" title "SALR Merged Density Browser"
        } else {
            print "ERROR: No interactive terminal available (wxt, qt, or x11)"
            print "Your gnuplot installation only has file output terminals."
            print ""
            print "SOLUTION: Use the Python visualization instead:"
            print "  python3 scripts/plot_joint_heatmap.py output/"
            exit
        }
    }
}

set ticslevel 0
set palette defined (0 "#440154", 0.25 "#31688e", 0.50 "#35b779", 0.75 "#fde725", 1.0 "#ff4400")

# ── Helper to update parameter labels ─────────────────────────────────────
update_labels = "unset label 1; unset label 2; unset label 3; unset label 4; unset label 5; unset label 6; unset label 7; unset label 8; unset label 9; " \
. "if (param_exists && show_params) { " \
.   "set label 1 sprintf('Grid: %.1fx%.1f  %dx%d\ndx=%.3f  dy=%.3f  BC=%s', grid_Lx, grid_Ly, grid_nx, grid_ny, grid_dx, grid_dy, grid_bc) at screen 0.01, screen 0.92 font 'Sans:Bold,12' tc rgb '#111111' front; " \
.   "set label 2 sprintf('Physics:\nT=%.2f  {/Symbol r}_{1}=%.2f\n{/Symbol r}_{2}=%.2f  r_{c}=%.1f', phys_T, phys_rho1, phys_rho2, phys_rc) at screen 0.01, screen 0.80 font 'Sans:Bold,12' tc rgb '#111111' front; " \
.   "set label 3 sprintf('Solver:\nxi1=%.4f  xi2=%.4f\ntol=%.1e  max=%d', solv_xi1, solv_xi2, solv_tol, solv_max) at screen 0.01, screen 0.63 font 'Sans:Bold,12' tc rgb '#111111' front; " \
.   "set label 4 sprintf('A_{11}=%.2f,%.2f,%.2f\n{/Symbol a}_{11}=%.3f,%.3f,%.3f', A11_1,A11_2,A11_3,a11_1,a11_2,a11_3) at screen 0.01, screen 0.47 font 'Sans:Bold,10' tc rgb '#333333' front; " \
.   "set label 5 sprintf('A_{12}=%.2f,%.2f,%.2f\n{/Symbol a}_{12}=%.3f,%.3f,%.3f', A12_1,A12_2,A12_3,a12_1,a12_2,a12_3) at screen 0.01, screen 0.34 font 'Sans:Bold,10' tc rgb '#333333' front; " \
.   "set label 6 sprintf('A_{22}=%.2f,%.2f,%.2f\n{/Symbol a}_{22}=%.3f,%.3f,%.3f', A22_1,A22_2,A22_3,a22_1,a22_2,a22_3) at screen 0.01, screen 0.21 font 'Sans:Bold,10' tc rgb '#333333' front; " \
. "}"

# ── Draw macro: 1×2 layout (3D scatter + 2D top-down view) ────────────────────────
DRAW = \
  "f1=word(files1,idx); f2=word(files2,idx); " \
. "fmix='< paste '.f1.' '.f2; " \
. "lbl=(strstrt(f1,'final')>0?'FINAL':sprintf('iter %d',int(real(system(\"echo '\" .f1. \"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "cmstr=(clip_mode?'clipped':'auto'); " \
. "eval update_labels; " \
. "set multiplot title sprintf('Frame %d/%d  --  %s    colour:%s    n/p±1  ]/[±10  N/P±100  f/l  c:clip  i:info  R:rotate  q:quit',idx,N,lbl,cmstr) font 'Sans:Bold,14'; " \
. "set origin 0.19,0.03; set size 0.37,0.93; " \
. "set xlabel 'x' font 'Sans:Bold,14'; set ylabel 'y' font 'Sans:Bold,14'; set zlabel '{/Symbol r}' rotate font 'Sans:Bold,14'; set ticslevel 0; set key top right font 'Sans:Bold,14'; " \
. "set view 60,30; " \
. "set title '3D Scatter (Purple=SALR, Green=Solvent)' font 'Sans:Bold,14'; " \
. "if (clip_mode) { set cbrange [0:cb_mix_clip] } else { set cbrange [*:*] }; " \
. "set style fill transparent solid 0.15 noborder; psize1=0.7; psize2=0.5; " \
. "splot f1 u 1:2:(\$3 > 1e-6 ? \$3 : 1/0) w p pt 7 ps psize1 lc rgb 0xBF8B008B title 'SALR', " \
. "      f2 u 1:2:(\$3 > 1e-6 ? \$3 : 1/0) w p pt 7 ps psize2 lc rgb 0xBF2E8B57 title 'Solvent'; " \
. "set origin 0.58,0.03; set size 0.40,0.93; " \
. "unset zlabel; unset view; unset key; set xlabel 'x' font 'Sans:Bold,14'; set ylabel 'y' font 'Sans:Bold,14'; " \
. "set title '2D Heatmap  (+1=SALR purple, -1=Solvent green)' font 'Sans:Bold,14'; " \
. "set cbrange [-1:1]; " \
. "set palette defined (-1 '#2E8B57', -0.1 '#112211', 0 '#111111', 0.1 '#221122', 1 '#8B008B'); " \
. "set style fill solid 1.0 noborder; " \
. "set colorbox; unset pm3d; set pm3d map; " \
. "splot '< paste '.f1.' '.f2 u 1:2:((\$3-\$6)/(\$3+\$6+1e-9)) w pm3d notitle; " \
. "unset pm3d; set palette defined (0 '#440154', 0.25 '#31688e', 0.50 '#35b779', 0.75 '#fde725', 1.0 '#ff4400'); " \
. "unset key; unset multiplot; unset origin; unset size; set view 60,30; set ticslevel 0; set zlabel '{/Symbol r}' rotate; set style fill solid 1.0 noborder; set cbrange [*:*]"

# ── Rotation mode (full-window 3D with both species) ────────────────────────
DRAW_ROT = \
  "f1=word(files1,idx); f2=word(files2,idx);" \
. "lbl=(strstrt(f1,'final')>0?'FINAL':sprintf('iter %d',int(real(system(\"echo '\" .f1. \"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "set title sprintf('[ROTATE MODE]  Frame %d/%d  %s    click-drag to rotate    any key -> browser',idx,N,lbl) font 'Sans:Bold,14';" \
. "set xlabel 'x' font 'Sans:Bold,14'; set ylabel 'y' font 'Sans:Bold,14'; set zlabel '{/Symbol r}' rotate font 'Sans:Bold,14'; set ticslevel 0; set key top right font 'Sans:Bold,14';" \
. "if (clip_mode) { set cbrange [0:cb_mix_clip] } else { set cbrange [*:*] };" \
. "set style fill transparent solid 0.15 noborder; psize1=0.7; psize2=0.5;" \
. "splot f1 u 1:2:(\$3 > 1e-6 ? \$3 : 1/0) w p pt 7 ps psize1 lc rgb 0xBF8B008B title 'SALR', " \
. "      f2 u 1:2:(\$3 > 1e-6 ? \$3 : 1/0) w p pt 7 ps psize2 lc rgb 0xBF2E8B57 title 'Solvent';" \
. "unset key; set style fill solid 1.0 noborder"

# ── Event loop ────────────────────────────────────────────────────────────
rot_mode = 0
idx = 1
eval DRAW

goto_n  = 0
has_dig = 0
kc = 0
ch = ""

running = 1
while (running) {
    pause mouse any

    if (exists("MOUSE_KEY")) { kc = int(MOUSE_KEY) } else { kc = 0 }
    if (exists("MOUSE_CHAR")) { ch = MOUSE_CHAR } else { ch = "" }

    # Mouse event filter: skip mouse-only events (rotation/pan handled by gnuplot)
    if (ch eq "" && kc < 65000) {
        # no-op: mouse event
    } else {

    # ── In rotation mode any keypress returns to browser ─────────────────
    if (rot_mode) {
        rot_mode = 0
        eval DRAW
    } else {

    # ── Digit accumulator and navigation (browser mode only) ─────────────
    if (strlen(ch) == 1 && strstrt("0123456789", ch) > 0) {
        goto_n  = goto_n * 10 + int(real(ch))
        has_dig = 1
        print sprintf("goto> %d_  (press Enter to jump, any other key cancels)", goto_n)
    } else {

        # Enter key: execute the accumulated goto
        if (kc == 65293 && has_dig) {
            idx = goto_n < 1 ? 1 : goto_n
            idx = idx > N ? N : idx
            print sprintf("jumped to frame %d", idx)
            eval DRAW
        }
        # Any non-digit resets the accumulator
        goto_n  = 0
        has_dig = 0

        # ── Navigation ───────────────────────────────────────────────────
        # +1 frame
        if (kc == 65363 || ch eq "n") {
            idx = (idx < N) ? idx+1 : N
            eval DRAW
        }
        # -1 frame
        if (kc == 65361 || ch eq "p") {
            idx = (idx > 1) ? idx-1 : 1
            eval DRAW
        }
        # +10 frames  (Page Down or ])
        if (kc == 65366 || ch eq "]") {
            idx = (idx+10 <= N) ? idx+10 : N
            eval DRAW
        }
        # -10 frames  (Page Up or [)
        if (kc == 65365 || ch eq "[") {
            idx = (idx-10 >= 1) ? idx-10 : 1
            eval DRAW
        }
        # +100 frames  (Shift-N or })
        if (ch eq "N" || ch eq "}") {
            idx = (idx+100 <= N) ? idx+100 : N
            eval DRAW
        }
        # -100 frames  (Shift-P or {)
        if (ch eq "P" || ch eq "{") {
            idx = (idx-100 >= 1) ? idx-100 : 1
            eval DRAW
        }
        # first frame  (Home or f)
        if (kc == 65360 || ch eq "f") {
            idx = 1
            eval DRAW
        }
        # last frame  (End or l)
        if (kc == 65367 || ch eq "l") {
            idx = N
            eval DRAW
        }
        # toggle colour clipping
        if (ch eq "c") {
            clip_mode = 1 - clip_mode
            eval DRAW
        }
        # toggle parameter info display
        if (ch eq "i") {
            show_params = 1 - show_params
            eval DRAW
        }
        # enter rotation mode
        if (ch eq "R" || ch eq "r") {
            rot_mode = 1
            eval DRAW_ROT
        }
        # quit
        if (ch eq "q" || ch eq "Q") {
            running = 0
        }
      }  # end digit else (navigation block)
    }  # end browser-mode block (rot_mode else)
    }  # end mouse-event filter (mouse-only else)
}
