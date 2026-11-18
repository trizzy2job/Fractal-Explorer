"""Optional GUI regression check (requires a graphical desktop / Xvfb on Linux)."""
import sys
from pathlib import Path
import tkinter as tk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cantor_explorer import CantorExplorer


def run():
    root = tk.Tk()
    app = CantorExplorer(root)
    root.update()
    assert [a.get() for a in app.axis_vars] == ['Re z1', 'Re z2', 'Im z1']
    assert len(app.rotation_rows) == 3
    assert not app.camera_row.winfo_manager()

    # A slider triggers a real asynchronous redraw without regenerating points.
    original_points = app.result.points
    before, _ = app._projection()
    redraws = [0]
    original_render = app.render

    def wrapped_render():
        redraws[0] += 1
        return original_render()

    app.render = wrapped_render
    app.rotation_rows[0]['slider'].set(46.5)
    root.after(130, root.quit)
    root.mainloop()
    after, _ = app._projection()
    assert redraws[0] >= 1, 'Slider movement did not automatically render.'
    assert not np.allclose(before, after), 'Slider did not change projection.'
    assert app.result.points is original_points, 'View changes must reuse point data.'

    app.view_var.set('3D')
    app.render()
    root.update()
    assert app.camera_row.winfo_manager() == 'pack'
    app.elev_scale.set(33)
    root.update()
    assert app.axes.elev == 33
    app._add_rotation()
    assert len(app.rotation_rows) == 4
    app._remove_rotation(1)
    assert len(app.rotation_rows) == 3
    app.rotation_rows[0]['second'].set('Im z2')
    app._rotation_plane_changed(0)
    assert app.rotations[0][1] == 3
    app._reset_rotation_angles()
    assert all(angle == 0 for i, j, angle in app.rotations)

    app.d_var.set(3)
    app._on_dimension_changed()
    app.generate()
    assert [a.get() for a in app.axis_vars] == ['Re z1', 'Re z2', 'Re z3']
    assert len(app.rotation_rows) == 3
    app.d_var.set(1)
    app._on_dimension_changed()
    app.generate()
    assert [a.get() for a in app.axis_vars] == ['Re z1', 'Im z1', '0 (fixed)']
    assert len(app.rotation_rows) == 1
    root.destroy()
    print('GUI smoke test passed: automatic redraw, 1D/2D/3D defaults, planes, and camera.')


if __name__ == '__main__':
    run()
