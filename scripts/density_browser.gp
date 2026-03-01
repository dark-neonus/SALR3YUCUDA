#!/usr/bin/env gnuplot
#
# density_browser.gp — Interactive frame-by-frame density browser
#
# Usage (from project root):
#   gnuplot scripts/density_browser.gp
#   gnuplot -e "output_dir='output'" scripts/density_browser.gp
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
#   q                     quit
#   v                     cycle view mode: scatter+heat | scatter only | heat only
#
# LAYOUT (default)
#   Top row:    3D scatter  rho_1 | rho_2 | rho_1+rho_2
#   Bottom row: 2D heatmap  rho_1 | rho_2 | rho_1+rho_2
#
# COLOUR MODES
#   Clipped (default): cbrange [0, 3*mean(rho)] — dilute background is
#                      clearly visible; cluster peak saturates to red.
#   Auto:             cbrange [*:*] — shows exact peak values but the
#                     dilute background appears near-black / "near-zero".
#
# NOTE on the "near-zero floor" appearance:
#   SALR cluster systems at low T have a dense cluster phase coexisting with
#   a dilute interstitial fluid.  During early iterations the background
#   density is ~0.15-0.25 (NOT zero) while cluster seeds reach rho~20-30.
#   Using auto colour makes the background look black.  Press 'c' to switch
#   to clipped mode to see the background structure clearly.
# ──────────────────────────────────────────────────────────────────────────

if (!exists("output_dir")) { output_dir = "output" }

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
# CB_SCALE * mean(rho) sets the upper end of the clipped colour range.
# At CB_SCALE=3 the bulk fluid (~mean) maps to the lower third of the ramp,
# making spatial inhomogeneity clearly visible even when clusters reach 20x mean.
CB_SCALE  = 3.0
rho1_mean = real(system("awk 'NR>1{s+=$3;n++}END{printf \"%.6f\",s/n}' ".word(files1,1)))
rho2_mean = real(system("awk 'NR>1{s+=$3;n++}END{printf \"%.6f\",s/n}' ".word(files2,1)))
cb1_clip   = CB_SCALE * rho1_mean
cb2_clip   = CB_SCALE * rho2_mean
cb_mix_clip = CB_SCALE * (rho1_mean + rho2_mean)
print sprintf("  rho1 mean=%.4f  colour clip=[0,%.3f]", rho1_mean, cb1_clip)
print sprintf("  rho2 mean=%.4f  colour clip=[0,%.3f]", rho2_mean, cb2_clip)
print sprintf("  rho_mix mean=%.4f  colour clip=[0,%.3f]", rho1_mean+rho2_mean, cb_mix_clip)

# clip_mode=1 (default) = fixed cbrange; clip_mode=0 = auto full range
clip_mode = 1

# ── Terminal & appearance ──────────────────────────────────────────────────
# Interactive terminals: try x11, wxt, qt in order of preference
# If none work, this script requires gnuplot with interactive terminal support
if (strstrt(GPVAL_TERMINALS, "x11")) {
    set terminal x11 size 2100,900 enhanced font "Sans,10" title "SALR Density Browser"
} else {
    if (strstrt(GPVAL_TERMINALS, "wxt")) {
        set terminal wxt size 2100,900 enhanced font "Sans,10" title "SALR Density Browser"
    } else {
        if (strstrt(GPVAL_TERMINALS, "qt")) {
            set terminal qt size 2100,900 enhanced font "Sans,10" title "SALR Density Browser"
        } else {
            print "ERROR: No interactive terminal available (x11, wxt, or qt)"
            print "Your gnuplot installation only has file output terminals."
            print ""
            print "SOLUTION: Use the Python visualization instead:"
            print "  python3 scripts/plot_joint_heatmap.py output/"
            exit
        }
    }
}
set xlabel "x" font ",10"
set ylabel "y" font ",10"
set zlabel "{/Symbol r}" rotate font ",10"
set ticslevel 0
set key off
set palette defined (0 "#440154", 0.25 "#31688e", 0.50 "#35b779", 0.75 "#fde725", 1.0 "#ff4400")

# ── Draw macro (browser: 2×3 multiplot) ─────────────────────────────────────
# Layout:   row 1 = 3D scatter   rho1 | rho2 | rho1+rho2
#           row 2 = 2D heatmap   rho1 | rho2 | rho1+rho2
#
# Heatmaps use `set pm3d map` (bilinear interpolation) instead of `w image`
# so the display is smooth even when the data has small residual pixel-scale
# variation.
#
# fmix = shell `paste f1 f2`; columns after paste: 1=x 2=y 3=rho1 4=x 5=y 6=rho2
DRAW = \
  "f1=word(files1,idx); f2=word(files2,idx);" \
. "fmix='< paste '.f1.' '.f2;" \
. "lbl=(strstrt(f1,'final')>0?'FINAL':sprintf('iter %d',int(real(system(\"echo '\" .f1. \"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "cmstr=(clip_mode?'clipped':'auto');" \
. "set multiplot layout 2,3 title sprintf('Frame %d/%d  --  %s  colour:%s   n/p+/-1  ]/[+/-10  N/P+/-100  f/l  0-9+Entr goto  c colour  R rotate  q quit',idx,N,lbl,cmstr) font ',10';" \
. "set xlabel 'x'; set ylabel 'y'; set zlabel '{/Symbol r}' rotate; set ticslevel 0; set view 60,30;" \
. "set title '{/Symbol r}_1   scatter' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb1_clip] } else { set cbrange [*:*] };" \
. "splot f1 u 1:2:(\$3 > rho1_mean*1.2 ? \$3 : 1/0):3 w p pt 7 ps 0.4 lc palette notitle;" \
. "set title '{/Symbol r}_2   scatter' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb2_clip] } else { set cbrange [*:*] };" \
. "splot f2 u 1:2:(\$3 > rho2_mean*1.2 ? \$3 : 1/0):3 w p pt 7 ps 0.7 lc palette notitle;" \
. "set title '{/Symbol r}_1+{/Symbol r}_2   scatter' font ',11'; set key top right font ',8';" \
. "if (clip_mode) { set cbrange [0:cb_mix_clip] } else { set cbrange [*:*] };" \
. "splot f1 u 1:2:(\$3 > rho1_mean*1.2 ? \$3 : 1/0) w p pt 7 ps 0.7 lc rgb '#8B008B' title 'SALR', f2 u 1:2:(\$3 > rho2_mean*1.2 ? \$3 : 1/0) w p pt 7 ps 0.4 lc rgb '#2E8B57' title 'Solvent';" \
. "unset key; unset zlabel; unset view; set xlabel 'x'; set ylabel 'y'; set pm3d map;" \
. "set title '{/Symbol r}_1   heatmap' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb1_clip] } else { set cbrange [*:*] };" \
. "splot f1 u 1:2:3 notitle;" \
. "set title '{/Symbol r}_2   heatmap' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb2_clip] } else { set cbrange [*:*] };" \
. "splot f2 u 1:2:3 notitle;" \
. "set title '{/Symbol r}_1+{/Symbol r}_2   heatmap' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb_mix_clip] } else { set cbrange [*:*] };" \
. "splot fmix u 1:2:(\$3+\$6) notitle;" \
. "unset pm3d; unset multiplot; set view 60,30; set ticslevel 0; set zlabel '{/Symbol r}' rotate"

# ── Rotation mode (full-window 3D) ──────────────────────────────────────────
# Draws all three species in a single full-window splot so the user can freely
# click-and-drag to rotate.  Any keyboard press returns to the browser.
DRAW_ROT = \
  "f1=word(files1,idx); f2=word(files2,idx);" \
. "fmix='< paste '.f1.' '.f2;" \
. "lbl=(strstrt(f1,'final')>0?'FINAL':sprintf('iter %d',int(real(system(\"echo '\" .f1. \"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "set title sprintf('[ROTATE MODE]  Frame %d/%d  %s    any key -> browser',idx,N,lbl) font ',11';" \
. "set xlabel 'x'; set ylabel 'y'; set zlabel '{/Symbol r}' rotate; set ticslevel 0; set key top right;" \
. "if (clip_mode) { set cbrange [0:cb_mix_clip] } else { set cbrange [*:*] };" \
. "splot f1 u 1:2:3:3 w p pt 7 ps 0.40 lc palette title '{/Symbol r}_1'," \
. "      f2 u 1:2:3:3 w p pt 6 ps 0.40 lc palette title '{/Symbol r}_2';" \
. "unset key"

# ── Event loop ────────────────────────────────────────────────────────────
# `pause mouse any` (not `keypress`) lets mouse events return the pause so
# the Qt window stays responsive.  Mouse events (button press/release, motion,
# scroll wheel) all have MOUSE_CHAR=="" and MOUSE_KEY < 65000 — these are
# filtered to no-ops so gnuplot can handle rotation/pan internally without the
# browser redrawing and resetting the view.  Real keyboard special keys
# (arrows, Home, End, PageUp/Down, Return) have kc >= 65000 and still work.
#
# To rotate a 3D scatter panel: simply click-and-drag it. The browser stays
# out of the way. Press any letter/nav key to resume frame navigation
# (which calls DRAW and resets the view to the full 2×3 layout).
#
# 'R' enters an explicit rotation mode: full-window splot of rho1+rho2,
# free rotation.  Any keypress in rotation mode returns to the 2×3 browser.
#
# MOUSE_KEY integers (X11 keysyms):
#   65361=Left  65363=Right  65360=Home  65367=End
#   65365=PageUp  65366=PageDown  65293=Return
rot_mode = 0   # 0 = browser, 1 = full-window rotation view
idx = 1
eval DRAW

goto_n  = 0   # digit accumulator for goto-frame
has_dig = 0   # whether any digit has been typed since last non-digit

kc = 0        # initialize keyboard code variable
ch = ""       # initialize character variable

running = 1
while (running) {
    pause mouse any

    if (exists("MOUSE_KEY")) { kc = int(MOUSE_KEY) } else { kc = 0 }
    if (exists("MOUSE_CHAR")) { ch = MOUSE_CHAR } else { ch = "" }

    # Mouse event filter: skip all mouse-device events so gnuplot can handle
    # rotation/pan internally without re-drawing.
    #
    # Mouse motion:          MOUSE_CHAR="" MOUSE_KEY=0
    # Mouse button press:    MOUSE_CHAR="" MOUSE_KEY=1/2/3
    # Mouse scroll wheel:    MOUSE_CHAR="" MOUSE_KEY=4/5
    # All these have ch=="" AND kc < 65000.
    #
    # Real keyboard special keys (arrows, Home, End, PageUp/Down, Return)
    # all have kc >= 65000 (X11 keysyms), so they still pass through.
    # Printable keys always have ch != "" so they pass regardless.
    if (ch eq "" && kc < 65000) {
        # no-op: gnuplot already handled the mouse event (rotation/pan)
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
