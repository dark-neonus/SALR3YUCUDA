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
_iter1 = system("ls -v ".output_dir."/density_species1_iter_*.dat 2>/dev/null")
_iter2 = system("ls -v ".output_dir."/density_species2_iter_*.dat 2>/dev/null")
_fin1  = system("test -f ".output_dir."/density_species1_final.dat && echo ".output_dir."/density_species1_final.dat || true")
_fin2  = system("test -f ".output_dir."/density_species2_final.dat && echo ".output_dir."/density_species2_final.dat || true")

files1 = system("echo '".(_iter1).(_fin1 eq "" ? "" : " "._fin1)."' | xargs")
files2 = system("echo '".(_iter2).(_fin2 eq "" ? "" : " "._fin2)."' | xargs")

N = words(files1)
if (N == 0) {
    print "ERROR: no density files found in '".output_dir."/'."
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
cb1_clip  = CB_SCALE * rho1_mean
cb2_clip  = CB_SCALE * rho2_mean
print sprintf("  rho1 mean=%.4f  colour clip=[0,%.3f]", rho1_mean, cb1_clip)
print sprintf("  rho2 mean=%.4f  colour clip=[0,%.3f]", rho2_mean, cb2_clip)

# clip_mode=1 (default) = fixed cbrange; clip_mode=0 = auto full range
clip_mode = 1

# ── Terminal & appearance ──────────────────────────────────────────────────
set terminal qt size 1450,720 enhanced font "Sans,10" title "SALR Density Browser"
set xlabel "x" font ",10"
set ylabel "y" font ",10"
set zlabel "{/Symbol r}" rotate font ",10"
set ticslevel 0
set key off
set palette defined (0 "#440154", 0.25 "#31688e", 0.50 "#35b779", 0.75 "#fde725", 1.0 "#ff4400")

# ── Draw macro ────────────────────────────────────────────────────────────
# Variables read at eval time: idx, files1, files2, N, clip_mode, cb1_clip, cb2_clip
DRAW = \
  "f1=word(files1,idx); f2=word(files2,idx);" \
. "lbl=(strstrt(f1,'final')>0?'FINAL':sprintf('iter %d',int(real(system(\"echo '\" .f1. \"' | grep -oP '(?<=iter_)[0-9]+'\")))));" \
. "cmstr=(clip_mode?sprintf('clipped [0,%.2g/%.2g]',cb1_clip,cb2_clip):'auto');" \
. "set multiplot layout 1,2 title sprintf('Frame %d/%d  --  %s   colour:%s   n/p +/-1   ]/[ +/-10   N/P +/-100   f/l first/last   0-9+Entr goto   c colour   q quit',idx,N,lbl,cmstr) font ',10';" \
. "set title '{/Symbol r}_1   species 1' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb1_clip] } else { set cbrange [*:*] };" \
. "splot f1 u 1:2:3:3 w p pt 7 ps 0.45 lc palette notitle;" \
. "set title '{/Symbol r}_2   species 2' font ',11';" \
. "if (clip_mode) { set cbrange [0:cb2_clip] } else { set cbrange [*:*] };" \
. "splot f2 u 1:2:3:3 w p pt 7 ps 0.45 lc palette notitle;" \
. "unset multiplot"

# ── Event loop ────────────────────────────────────────────────────────────
# MOUSE_CHAR  = string for printable keys  ("n", "p", "q", "]", "[", ...)
# MOUSE_KEY   = X11 keysym integer for special keys:
#   65361=Left  65363=Right  65360=Home  65367=End
#   65365=PageUp  65366=PageDown  65293=Return
idx    = 1
eval DRAW

goto_n  = 0   # digit accumulator for goto-frame
has_dig = 0   # whether any digit has been typed since last non-digit

running = 1
while (running) {
    pause mouse keypress

    kc = int(MOUSE_KEY)
    ch = MOUSE_CHAR

    # ── Digit accumulator: type a number, then press Enter to jump ─────────
    # strstrt("0123456789", ch) > 0  is true iff ch is exactly one digit.
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
        # quit
        if (ch eq "q" || ch eq "Q") {
            running = 0
        }
    }
}
