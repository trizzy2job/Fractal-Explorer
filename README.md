# Complex Cantor Explorer

Interactive desktop explorer for finite-stage self-affine constructions in complex dimensions `1 <= d <= 6`. It accepts an arbitrary **invertible** `d x d` complex matrix and an explicit finite alphabet of complex vectors. It displays point approximations and projected geometric cylinder cells, rotates selected coordinate planes of the realified `2d`-dimensional space, and exports PNG or SVG.

## Install and run

**Windows (Python 3.10 or newer):** Extract the ZIP, open PowerShell in the extracted folder, then run:

```powershell
py -m pip install -r requirements.txt
py cantor_explorer.py
```

Alternatively double-click `install_and_run_windows.bat` after installing Python from python.org (enable **Add Python to PATH**).

**macOS / Linux (Python 3.10 or newer):**

```bash
python3 -m pip install -r requirements.txt
python3 cantor_explorer.py
```

Tkinter normally ships with Windows/macOS Python. On some Linux distributions, install it separately (for example `sudo apt install python3-tk`). The interface needs a desktop display; it is not a command-line renderer.

## Mathematical model

Given an invertible matrix `M in C^(d x d)` and a finite alphabet `A subset C^d`, the program computes the *multiset of word locations*

```
C_0 = {0}
C_k = {M^(-1) a_1 + M^(-2) a_2 + ... + M^(-k) a_k : a_i in A}.
```

It also uses the equivalent recurrence `C_(k+1) = M^(-1)(C_k + A)` to enumerate points without storing all earlier stages.

For `M` **strictly expanding** (all complex eigenvalues have modulus greater than 1), the iterated function system `f_a(x) = M^(-1)(x+a)` has a unique nonempty compact attractor `K`. This does not, by itself, ensure that `K` is totally disconnected, perfect, or a Cantor set. The application intentionally permits non-expanding invertible matrices for studying *finite stages*, but explicitly warns that the compact-attractor guarantee may fail.

### Geometric cells

Every word at stage `k` defines a cylinder `t_w + M^(-k)K`. When expanding, the program finds a containing axis-aligned real box `Q` with `K subset Q` and displays the larger **covering cell**

```
t_w + M^(-k) Q.
```

The projection of this cell onto 2D or 3D is computed as a convex zonotope. With the axis-aligned interval bound `rho(|realify(M^(-1))|) < 1`, the program solves for an invariant box. Otherwise, it uses a conservative power-norm bound if one can be certified numerically. If expansion or certification fails, it displays *reference cells* instead, clearly labeled as such. These cells should not be confused with the actual fractal cylinder sets when `Q` is a strict superset of `K`.

Cell subsets are sometimes drawn to keep rendering responsive. The status line gives the exact word count and the displayed count.

## Entering matrices and alphabets

The editor accepts one row per line with **comma-separated numerical expressions**. Complex numbers use Python's `j` notation (`2+0.7j`, `-1j`, `3`, etc.). You can include `#` comments. The parser uses `ast.literal_eval`, not `eval`, so arbitrary Python expressions and functions are not permitted.

**Classic middle-third Cantor set:** Set `d = 1`, `k = 6`, matrix input:

```
3
```

Alphabet input:

```
0
2
```

This displays points in `C` on the real axis. Selecting **Cells** draws the corresponding containing intervals, which in this example are exactly the standard Cantor construction intervals.

**Complex two-dimensional example:** Set `d = 2`, matrix:

```
3, 0.4+0.2j
0, 3
```

Alphabet:

```
0, 0
2, 0
0, 2+0.7j
2+0.4j, 2
```

The application starts with this example, at stage `k=4`.

For `d=6`, enter six matrix rows (each six coefficients) and alphabet rows with six coefficients. `Load example for current d` constructs a working example at any supported dimension.

### Changing dimension without losing custom input

Changing the **Complex dimension d** setting updates a field only if that field
still equals its last generated example. In particular:

* Untouched example matrix: resized to the new `d x d` default.
* Edited/custom matrix: preserved verbatim, even if the new dimension makes it invalid.
* Untouched example alphabet: resized to the new default digit vectors.
* Edited/custom alphabet: preserved verbatim independently of the matrix.
* Changing the **Iteration k** setting never changes matrix or alphabet text.

Clicking **Load example for current d** explicitly replaces **both** editors,
even if they contain custom data. If preserved custom entries no longer match
`d`, edit them yourself; **Validate inputs** or **Generate / regenerate** will
explain exactly which field needs correction.

### Input validation and audible errors

Use **Validate inputs** to check all parameters and the matrix without allocating
the finite-stage point cloud. **Generate / regenerate** performs the same checks
before starting any numerical work. Validation catches:

* Invalid dimensions or stages, or out-of-range point/cell budgets.
* Malformed numbers and incorrect matrix or alphabet row lengths.
* Singular or numerically singular matrices, tested using singular values.

Invalid input produces a dedicated, field-specific dialog with correction
instructions. On Windows it plays the current system error sound using
`MessageBeep(MB_ICONHAND)`; elsewhere it uses the Tk bell. Invalid input does
not replace the previous successfully generated point set. The SVD tolerance
is numerical: sufficiently ill-conditioned but mathematically invertible
matrices may be rejected as unsafe for double-precision inversion.

## Rotating and projecting

`C^d` is identified with `R^(2d)` using this real-coordinate order:

```
(Re z1, Re z2, ..., Re zd, Im z1, Im z2, ..., Im zd).
```

The **rotation sliders appear immediately beneath the graph**, next to the standard
Matplotlib navigation toolbar. Moving a slider automatically redraws the current
projection; there is no need to click **Generate / regenerate**.

1. **Ambient coordinate-plane rotations:** Each active plane has its own slider
   (from -180 to +180 degrees), together with two coordinate selectors. Three
   initial sliders rotate the first three real coordinates: `Re z1 / Re z2`,
   `Re z1 / Re z3`, `Re z2 / Re z3` for `d >= 3`. For `d = 2`, the third
   coordinate is `Im z1`; for `d = 1`, there is one slider for
   `Re z1 / Im z1`. Select any pair of real or imaginary coordinates to
   rotate in a different plane. **+ Add plane** inserts a new slider;
   **Reset angles** zeros the current sliders; **Default planes** restores
   the defaults. The small **x** on a row removes that rotation. Ambient
   rotations are applied in row order.
2. **Camera rotation for 3D:** Switching to 3D reveals *elevation* and
   *azimuth* sliders beneath the plane sliders. Dragging the plot with the
   mouse also rotates the camera. Camera controls do not change the actual
   coordinates or the ambient-space projection.

The initial 2D projection is `Re z1, Re z2` when `d >= 2`; the initial 3D
projection is `Re z1, Re z2, Re z3` when `d >= 3`. Missing real coordinates
are filled with `Im z1` (and then `0 (fixed)` for the third axis of `d = 1`).
All three plotted coordinates can also be selected independently from the
**X/Y/Z** controls in the left sidebar. Projection occurs after all ambient
rotations; it is an orthogonal coordinate projection into 2D or 3D.

The **Zoom** slider and mouse wheel change the graph scale. The Matplotlib
navigation toolbar offers pan, zoom, reset, and graph navigation. Rotation
sliders redraw from the cached point array rather than recomputing digit
words, making viewpoint changes considerably faster than a full new generation.

## Export and presets

* **Export PNG**: Saves the currently visible plot as a PNG (220 dpi).
* **Export SVG**: Saves the currently visible plot in SVG format. Point markers and geometric cells remain vector graphics; a high-budget scatter plot can therefore produce a large SVG file.
* **Save preset / Open preset**: Save or restore the matrix, alphabet, stage, axes, rotations, zoom, and display settings as JSON.

Exports reflect **the points/cells currently drawn**, not omitted exact words or sampled words beyond the selected budget. Zoom, projection, and camera orientation are preserved.

## Performance / memory management

- **Exact mode:** `m = |A|` digits, stage `k`, at most `m^k` words. If `m^k <= maximum stored points`, enumerate all words using one growing complex128 array, one previous-stage array, and one reused matrix multiplication. Stage intermediates are discarded immediately.
- **Sampling mode:** If `m^k` exceeds the point budget, sample exactly that many digit sequences *with replacement* (using a fixed reproducible PRNG seed). The sampler keeps two `N x d` complex128 buffers, employs `numpy.matmul(..., out=...)`, and adds random digits in small blocks; it never forms an `N x k x d` tensor.
- **Projection:** Rather than materializing a permanent `N x 2d` real-coordinate array, multiply the real and imaginary parts separately by the relevant parts of the real projection matrix.
- **Geometry:** Construct a projected zonotope hull just once per projection (up to `2^(2d) <= 4096` candidate vertices), then translate a *single batched collection* to all displayed cell centers.
- **Rendering throttles:** At most 120,000 markers in 2D or 25,000 markers in 3D; at most 1,500 cells in 2D or 90 in 3D, even if you request more. A 55 ms coalescing redraw throttle updates continuously during dragging, without regenerating the finite-stage point array. 3D Matplotlib rendering is CPU-based and is not a GPU particle renderer.
- **Storage:** A complex128 point array needs `16 * N * d` bytes, excluding drawing and working arrays. At the default 120,000-point budget with `d=6`, the *points alone* use about 11 MiB. Python, plotting, and allocation workspaces add overhead. The user-adjustable maximum is 2,000,000 points; increasing it can consume several hundred MiB of RAM.

These choices were informed by inspection of the installed Matplotlib 3.10.8 implementations of `Axes3D.scatter`, `Poly3DCollection.do_3d_projection`, and `Collection.set_offsets`. In particular, Matplotlib's 3D polygons undergo per-face projection and Python-side depth sorting; this is the reason for batching faces and capping 3D cell counts. The code also uses NumPy's ufunc `out=` and `matmul(..., out=...)` facilities to reuse buffers.

## Testing

From the extracted folder (with `pytest` installed):

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests cover exact complex digit expansions, a middle-thirds construction, dimension changes at stage zero, safe input parsing, singular matrices, deterministic sampling, plane rotation, and both covering-cell and lower-rank projection geometry. The GUI was also smoke-tested in a virtual desktop at `d=6`, stage `k=8`, including both 2D and 3D drawing and PNG/SVG export. The revised UI has additionally been tested for slider-driven automatic renders, default real-axis selection at `d=1,2,3`, dynamically adding/removing planes, and 3D camera controls. Run `python tests/gui_smoke.py` with a graphical display to repeat the UI check. Run `python tests/gui_validation.py` to test independent preservation of custom editors, stage changes, matrix singularity validation, and the custom error dialog.

## Limits and interpretation

Matrix inversion and all digit calculations use floating-point complex128, not exact arithmetic. Numerical accuracy can deteriorate for ill-conditioned matrices; a warning is shown for large condition numbers. The finite stage records words, **not necessarily distinct spatial points**; overlaps and repeated translations are not removed. Projections can cause additional collisions or obscure the topology of the true `2d`-dimensional set. The program does not try to establish total disconnectedness or Hausdorff dimension for arbitrary inputs.
