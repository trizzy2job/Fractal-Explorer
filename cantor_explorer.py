"""Interactive desktop explorer for finite-stage Cantor-type sets in C^d.

Run: python cantor_explorer.py
Requires: numpy, scipy, matplotlib; Tkinter is included in common Python installs.
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np

# Import TkAgg explicitly so the application behaves consistently regardless
# of the user's matplotlibrc/backend setting.
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from cantor_core import (
    MAX_COMPLEX_DIM, MAX_STAGE, generate_points, invariant_envelope,
    parse_alphabet, parse_matrix, project, projected_cell_geometry,
    projection_matrix,
)

BG = '#f4f6fa'
INK = '#21334d'
BLUE = '#2264a7'
ORANGE = '#e8a43e'


class InputValidationError(ValueError):
    """An input error with a field name and actionable correction guidance."""

    def __init__(self, field: str, detail: str, hint: str):
        super().__init__(detail)
        self.field = field
        self.hint = hint


def example_input(d: int) -> tuple[str, str]:
    if d == 1:
        return '3', '0\n2'
    M = np.eye(d, dtype=np.complex128) * 3
    M[0, 1] = 0.4 + 0.2j
    rows = []
    for row in M:
        rows.append(', '.join(_fmt_complex(v) for v in row))
    digits = np.zeros((4, d), dtype=np.complex128)
    digits[1, 0] = 2
    digits[2, 1] = 2 + 0.7j
    digits[3, 0] = 2 + 0.4j
    digits[3, 1] = 2
    return '\n'.join(rows), '\n'.join(', '.join(_fmt_complex(v) for v in row) for row in digits)


def _fmt_complex(z: complex) -> str:
    a, b = float(np.real(z)), float(np.imag(z))
    if abs(b) < 1e-14:
        return f'{a:g}'
    if abs(a) < 1e-14:
        return f'{b:g}j'
    return f'{a:g}{b:+g}j'


class CantorExplorer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Complex Cantor Explorer  |  Finite-stage self-affine sets')
        self.root.geometry('1430x900')
        self.root.minsize(1030, 640)
        self.root.configure(bg=BG)
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('TLabelFrame', background=BG, foreground=INK)
        style.configure('TLabelFrame.Label', background=BG, foreground=INK)
        style.configure('TButton', padding=(7, 5))
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Small.TLabel', font=('Segoe UI', 9))

        self.d_var = tk.IntVar(value=2)
        self.stage_var = tk.IntVar(value=4)
        self.max_points_var = tk.StringVar(value='120000')
        self.max_cells_var = tk.StringVar(value='250')
        self.view_var = tk.StringVar(value='2D')
        self.display_var = tk.StringVar(value='Points')
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_label = tk.StringVar(value='1.00x')
        self.elev_var = tk.DoubleVar(value=24)
        self.azim_var = tk.DoubleVar(value=-60)
        self.axis_vars = [tk.StringVar() for _ in range(3)]
        self.rotations: list[tuple[int, int, float]] = []
        self.rotation_rows: list[dict] = []
        self._rebuilding_rotations = False
        self.result = None
        self.M = None
        self.alphabet = None
        self.envelope = None
        self.axes = None
        self._center = None
        self._span = 1.0
        self._render_job = None
        self._syncing = False
        self._setting_dim = False
        self._valid_dimension = 2
        # These are the most recently offered defaults, not the user's edits.
        # Track the editors independently so customizing one does not freeze
        # the other on subsequent dimension changes.
        self._default_matrix_text = ''
        self._default_alphabet_text = ''
        self.status_var = tk.StringVar(value='Ready.')
        self.warning_var = tk.StringVar(value='')

        self._make_layout()
        self._load_example()
        self._set_axis_choices(reset=True)
        self._restore_default_rotations(render=False)
        self._make_figure()
        self.generate()

    def _make_layout(self):
        whole = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        whole.pack(fill='both', expand=True)
        sidebar = ttk.Frame(whole, width=420)
        viewer = ttk.Frame(whole)
        whole.add(sidebar, weight=0)
        whole.add(viewer, weight=1)
        self.viewer = viewer

        scroll_container = tk.Canvas(sidebar, bg=BG, highlightthickness=0, width=395)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=scroll_container.yview)
        scroll_container.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        scroll_container.pack(side='left', fill='both', expand=True)
        self.control_panel = ttk.Frame(scroll_container, padding=(12, 10))
        win = scroll_container.create_window((0, 0), window=self.control_panel, anchor='nw')
        self.control_panel.bind('<Configure>', lambda _: scroll_container.configure(scrollregion=scroll_container.bbox('all')))
        scroll_container.bind('<Configure>', lambda ev: scroll_container.itemconfigure(win, width=ev.width))
        scroll_container.bind('<Enter>', lambda _: scroll_container.bind_all('<MouseWheel>', self._sidebar_wheel))
        scroll_container.bind('<Leave>', lambda _: scroll_container.unbind_all('<MouseWheel>'))
        self._sidebar_canvas = scroll_container

        panel = self.control_panel
        ttk.Label(panel, text='Complex Cantor Explorer', style='Heading.TLabel').pack(anchor='w')
        ttk.Label(panel, text='C_k = {sum M^(-j) a_j}   |   d <= 6', style='Small.TLabel').pack(anchor='w', pady=(3, 10))

        setting = ttk.LabelFrame(panel, text='1. Iteration and drawing', padding=9)
        setting.pack(fill='x', pady=5)
        self._spin_row(setting, 'Complex dimension d', self.d_var, 1, MAX_COMPLEX_DIM, self._on_dimension_changed)
        self._spin_row(setting, 'Iteration k', self.stage_var, 0, MAX_STAGE)
        self._entry_row(setting, 'Maximum stored points', self.max_points_var)
        self._entry_row(setting, 'Maximum drawn cells', self.max_cells_var)
        ttk.Label(setting, text='Above the point budget, sample digit words uniformly.', wraplength=345, style='Small.TLabel').pack(anchor='w', pady=3)

        editor = ttk.LabelFrame(panel, text='2. Matrix and alphabet', padding=9)
        editor.pack(fill='x', pady=5)
        ttk.Label(editor, text='M: d rows, d comma-separated complex entries').pack(anchor='w')
        self.matrix_text = tk.Text(editor, width=38, height=4, font=('Consolas', 10), undo=True, wrap='none')
        self.matrix_text.pack(fill='x', pady=(3, 7))
        ttk.Label(editor, text='A: one digit vector per row (use j for sqrt(-1))').pack(anchor='w')
        self.alpha_text = tk.Text(editor, width=38, height=5, font=('Consolas', 10), undo=True, wrap='none')
        self.alpha_text.pack(fill='x', pady=(3, 6))
        ttk.Button(editor, text='Load example for current d', command=self._load_example).pack(fill='x')
        ttk.Label(editor, text='Changing d preserves edited fields; untouched examples resize automatically.',
                  wraplength=345, style='Small.TLabel').pack(anchor='w', pady=(4, 0))

        view_frame = ttk.LabelFrame(panel, text='3. Visualization / projection', padding=9)
        view_frame.pack(fill='x', pady=5)
        self._combo_row(view_frame, 'View', self.view_var, ['2D', '3D'], self.schedule_render)
        self._combo_row(view_frame, 'Draw', self.display_var, ['Points', 'Cells', 'Both'], self.schedule_render)
        self.axis_combos = []
        for a in range(3):
            cb = self._combo_row(view_frame, ('X', 'Y', 'Z')[a] + ' coordinate', self.axis_vars[a], [], self.schedule_render)
            self.axis_combos.append(cb)
        ttk.Label(view_frame, text='Projection axes are selected after all stored rotations.', wraplength=340,
                  style='Small.TLabel').pack(anchor='w', pady=3)

        camera = ttk.LabelFrame(panel, text='4. Zoom', padding=9)
        camera.pack(fill='x', pady=5)
        ttk.Label(camera, text='Scroll over plot to zoom. Use toolbar for pan/zoom; drag in 3D to orbit.',
                  wraplength=340, style='Small.TLabel').pack(anchor='w')
        self.zoom_scale = tk.Scale(camera, from_=0.25, to=12, resolution=.05, orient='horizontal',
                                   variable=self.zoom_var, label='Zoom', showvalue=True,
                                   background=BG, highlightthickness=0,
                                   command=lambda _: self._apply_zoom())
        self.zoom_scale.pack(fill='x')

        actions = ttk.LabelFrame(panel, text='5. Actions', padding=9)
        actions.pack(fill='x', pady=5)
        ttk.Button(actions, text='Validate inputs', command=self.validate_inputs).pack(fill='x', pady=3)
        ttk.Button(actions, text='Generate / regenerate', command=self.generate).pack(fill='x', pady=3)
        r = ttk.Frame(actions)
        r.pack(fill='x')
        ttk.Button(r, text='Export PNG', command=lambda: self.export('png')).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(r, text='Export SVG', command=lambda: self.export('svg')).pack(side='left', expand=True, fill='x', padx=2)
        r = ttk.Frame(actions)
        r.pack(fill='x', pady=(4,0))
        ttk.Button(r, text='Save preset', command=self.save_preset).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(r, text='Open preset', command=self.load_preset).pack(side='left', expand=True, fill='x', padx=2)

        status = ttk.Frame(self.viewer, padding=(10, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var, wraplength=850, justify='left').pack(anchor='w')
        tk.Label(status, textvariable=self.warning_var, fg='#995215', bg=BG,
                 wraplength=850, justify='left').pack(anchor='w')

    def _make_figure(self):
        self._make_rotation_dock()
        self.figure = Figure(figsize=(8, 7), dpi=105, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viewer)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.viewer, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.canvas.mpl_connect('scroll_event', self._mouse_zoom)
        self.canvas.mpl_connect('button_release_event', self._mouse_release)

    @staticmethod
    def _spin_row(parent, label, var, minimum, maximum, command=None):
        r = ttk.Frame(parent)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label).pack(side='left')
        box = ttk.Spinbox(r, from_=minimum, to=maximum, textvariable=var, width=9,
                          command=command)
        box.pack(side='right')
        if command:
            box.bind('<FocusOut>', lambda _: command())
            box.bind('<Return>', lambda _: command())
        return box

    @staticmethod
    def _entry_row(parent, label, var):
        r = ttk.Frame(parent)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label).pack(side='left')
        ttk.Entry(r, textvariable=var, width=10).pack(side='right')

    @staticmethod
    def _combo_row(parent, label, var, values, callback):
        r = ttk.Frame(parent)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text=label).pack(side='left')
        cb = ttk.Combobox(r, values=values, textvariable=var, width=17, state='readonly')
        cb.pack(side='right')
        cb.bind('<<ComboboxSelected>>', lambda _: callback())
        return cb

    def _sidebar_wheel(self, event):
        # Tk on Windows reports delta in units of 120; macOS varies.
        amount = -1 if event.delta > 0 else 1
        self._sidebar_canvas.yview_scroll(amount * 3, 'units')

    def _on_dimension_changed(self):
        if self._setting_dim:
            return
        try:
            d = int(self.d_var.get())
        except (ValueError, tk.TclError) as exc:
            self._show_input_error('Dimension', f'Enter a whole number from 1 to {MAX_COMPLEX_DIM}.',
                                   'The previous dimension has been restored.')
            self.d_var.set(self._valid_dimension)
            return
        if not 1 <= d <= MAX_COMPLEX_DIM:
            self._show_input_error('Dimension', f'The dimension must be between 1 and {MAX_COMPLEX_DIM}.',
                                   'The previous dimension has been restored.')
            self.d_var.set(self._valid_dimension)
            return
        if d == self._valid_dimension:
            return

        self._setting_dim = True
        try:
            matrix_was_default, alphabet_was_default = self._follow_default_inputs(d)
            self._set_axis_choices(reset=True)
            self._restore_default_rotations(render=False)
            self._valid_dimension = d
        finally:
            self._setting_dim = False
        kept = [name for name, was_default in (
            ('matrix', matrix_was_default), ('alphabet', alphabet_was_default))
            if not was_default]
        note = ('Preserved custom ' + ' and '.join(kept) + '. ') if kept else 'Updated default inputs. '
        self.status_var.set(f'Dimension changed to d={d}. {note}Click Generate to update the point set.')
        self.warning_var.set('Custom inputs may need resizing to match the selected dimension.' if kept else '')

    @staticmethod
    def _editor_contents(editor: tk.Text) -> str:
        return editor.get('1.0', 'end-1c').strip()

    @staticmethod
    def _replace_editor(editor: tk.Text, text: str):
        editor.delete('1.0', 'end')
        editor.insert('1.0', text)

    def _follow_default_inputs(self, d: int) -> tuple[bool, bool]:
        """Resize only fields that still contain their *previous* example.

        Independent comparisons also respect edits to only the alphabet or
        only the matrix. Stage changes never touch either field.
        """
        matrix_is_default = self._editor_contents(self.matrix_text) == self._default_matrix_text.strip()
        alphabet_is_default = self._editor_contents(self.alpha_text) == self._default_alphabet_text.strip()
        matrix_default, alphabet_default = example_input(d)
        if matrix_is_default:
            self._replace_editor(self.matrix_text, matrix_default)
        if alphabet_is_default:
            self._replace_editor(self.alpha_text, alphabet_default)
        self._default_matrix_text = matrix_default
        self._default_alphabet_text = alphabet_default
        self.matrix_text.configure(height=max(3, min(7, d + 1)))
        return matrix_is_default, alphabet_is_default

    def _load_example(self):
        try:
            d = int(self.d_var.get())
        except (ValueError, tk.TclError):
            d = -1
        if not 1 <= d <= MAX_COMPLEX_DIM:
            self._show_input_error('Dimension', 'Choose a dimension between 1 and 6.',
                                   'Correct the dimension field, then load the example.')
            return
        m, a = example_input(d)
        self._replace_editor(self.matrix_text, m)
        self._replace_editor(self.alpha_text, a)
        self._default_matrix_text, self._default_alphabet_text = m, a
        self.matrix_text.configure(height=max(3, min(7, d + 1)))
        self._set_axis_choices(reset=False)

    def _axis_labels(self) -> list[str]:
        d = int(self.d_var.get())
        return [f'Re z{i+1}' for i in range(d)] + [f'Im z{i+1}' for i in range(d)]

    def _set_axis_choices(self, reset: bool):
        """Prefer Re z1, Re z2, Re z3 whenever these coordinates exist.

        For d=1, Y defaults to Im z1 and Z is fixed; for d=2, the
        fallback Z coordinate is Im z1. No artificial zero axis is used
        where a real coordinate is available.
        """
        names = self._axis_labels()
        d = int(self.d_var.get())
        defaults = [names[0], names[1] if d >= 2 else names[d],
                    names[2] if d >= 3 else (names[d] if d == 2 else '0 (fixed)')]
        for pos, cb in enumerate(getattr(self, 'axis_combos', [])):
            cb.configure(values=names + ['0 (fixed)'])
            if reset or self.axis_vars[pos].get() not in names + ['0 (fixed)']:
                self.axis_vars[pos].set(defaults[pos])

    def _make_rotation_dock(self):
        """Keep immediately adjustable ambient-space sliders beneath the plot."""
        dock = ttk.LabelFrame(self.viewer, text='Viewing rotations  |  R^(2d)', padding=(9, 5))
        dock.pack(side='bottom', fill='x', padx=8, pady=(3, 0))
        self.rotation_dock = dock

        header = ttk.Frame(dock)
        header.pack(fill='x', pady=(0, 3))
        ttk.Label(header, text='Drag a slider to reproject immediately.', style='Small.TLabel').pack(side='left')
        ttk.Button(header, text='Default planes', command=self._restore_default_rotations).pack(side='right', padx=2)
        ttk.Button(header, text='Reset angles', command=self._reset_rotation_angles).pack(side='right', padx=2)
        ttk.Button(header, text='+ Add plane', command=self._add_rotation).pack(side='right', padx=2)

        list_frame = ttk.Frame(dock)
        list_frame.pack(fill='x')
        self.rotation_canvas = tk.Canvas(list_frame, height=116, bg=BG, highlightthickness=0, borderwidth=0)
        rot_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.rotation_canvas.yview)
        self.rotation_canvas.configure(yscrollcommand=rot_scrollbar.set)
        rot_scrollbar.pack(side='right', fill='y')
        self.rotation_canvas.pack(side='left', fill='both', expand=True)
        self.rotation_items = ttk.Frame(self.rotation_canvas)
        win = self.rotation_canvas.create_window((0, 0), window=self.rotation_items, anchor='nw')
        self.rotation_items.bind('<Configure>', lambda _:
            self.rotation_canvas.configure(scrollregion=self.rotation_canvas.bbox('all')))
        self.rotation_canvas.bind('<Configure>', lambda ev:
            self.rotation_canvas.itemconfigure(win, width=ev.width))
        self._rebuild_rotation_rows()

        # 3D camera orientation belongs in the same viewing-control area.
        self.camera_row = ttk.Frame(dock)
        self.camera_row.pack(fill='x', pady=(3, 0))
        ttk.Label(self.camera_row, text='3D camera', width=10).pack(side='left')
        ttk.Label(self.camera_row, text='Elevation').pack(side='left')
        self.elev_scale = tk.Scale(self.camera_row, from_=-90, to=90, resolution=1,
                                   orient='horizontal', variable=self.elev_var,
                                   showvalue=True, background=BG, highlightthickness=0,
                                   command=lambda _: self._camera_changed(), length=160)
        self.elev_scale.pack(side='left', fill='x', expand=True)
        ttk.Label(self.camera_row, text='Azimuth').pack(side='left')
        self.azim_scale = tk.Scale(self.camera_row, from_=-180, to=180, resolution=1,
                                   orient='horizontal', variable=self.azim_var,
                                   showvalue=True, background=BG, highlightthickness=0,
                                   command=lambda _: self._camera_changed(), length=160)
        self.azim_scale.pack(side='left', fill='x', expand=True)
        self._update_camera_visibility()

    def _update_camera_visibility(self):
        if not hasattr(self, 'camera_row'):
            return
        if self.view_var.get() == '3D':
            if not self.camera_row.winfo_manager():
                self.camera_row.pack(fill='x', pady=(3, 0))
        else:
            self.camera_row.pack_forget()

    def _default_rotation_planes(self) -> list[tuple[int, int, float]]:
        # Three coordinate-plane rotations spanning the default 3D slice,
        # or one rotation if the ambient real dimension is only two.
        d = int(self.d_var.get())
        n = min(3, 2*d)
        return [(i, j, 0.0) for i in range(n) for j in range(i+1, n)]

    def _restore_default_rotations(self, render=True):
        self.rotations = self._default_rotation_planes()
        self._rebuild_rotation_rows()
        if render:
            self.schedule_render()

    def _rebuild_rotation_rows(self):
        if not hasattr(self, 'rotation_items'):
            return
        self._rebuilding_rotations = True
        for child in self.rotation_items.winfo_children():
            child.destroy()
        self.rotation_rows = []
        names = self._axis_labels()
        for index, (i, j, angle) in enumerate(self.rotations):
            row = ttk.Frame(self.rotation_items)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=f'{index + 1:>2}.', width=3).pack(side='left')
            first = tk.StringVar(value=names[i])
            second = tk.StringVar(value=names[j])
            first_cb = ttk.Combobox(row, width=10, state='readonly', values=names, textvariable=first)
            first_cb.pack(side='left', padx=(1, 2))
            ttk.Label(row, text='/').pack(side='left')
            second_cb = ttk.Combobox(row, width=10, state='readonly', values=names, textvariable=second)
            second_cb.pack(side='left', padx=(2, 7))
            angle_var = tk.DoubleVar(value=angle)
            value_label = ttk.Label(row, text=f'{angle:+.1f}°', width=9)
            remove_button = ttk.Button(row, text='×', width=3,
                                       command=lambda pos=index: self._remove_rotation(pos))
            remove_button.pack(side='right', padx=(3, 0))
            value_label.pack(side='right')
            slider = tk.Scale(row, from_=-180, to=180, resolution=0.5, orient='horizontal',
                              showvalue=False, variable=angle_var, background=BG,
                              highlightthickness=0, borderwidth=0, sliderlength=17,
                              command=lambda value, pos=index: self._rotation_slider_changed(pos, value))
            slider.pack(side='left', fill='x', expand=True)
            first_cb.bind('<<ComboboxSelected>>', lambda _, pos=index: self._rotation_plane_changed(pos))
            second_cb.bind('<<ComboboxSelected>>', lambda _, pos=index: self._rotation_plane_changed(pos))
            self.rotation_rows.append({'first': first, 'second': second, 'angle': angle_var,
                                       'label': value_label, 'slider': slider})
        if not self.rotations:
            ttk.Label(self.rotation_items,
                      text='No active planes. Click + Add plane to create a rotation slider.',
                      style='Small.TLabel').pack(anchor='w', pady=(8, 4))
        self._rebuilding_rotations = False

    def _rotation_plane_changed(self, index):
        if self._rebuilding_rotations or index >= len(self.rotations):
            return
        row = self.rotation_rows[index]
        names = self._axis_labels()
        i = names.index(row['first'].get())
        j = names.index(row['second'].get())
        if i == j:
            # Reject degenerate planes in place without opening a modal dialog.
            old_i, old_j, _ = self.rotations[index]
            row['first'].set(names[old_i])
            row['second'].set(names[old_j])
            self.warning_var.set('Rotation plane must use two distinct coordinates.')
            return
        self.rotations[index] = (i, j, self.rotations[index][2])
        self.schedule_render()

    def _rotation_slider_changed(self, index, value):
        if self._rebuilding_rotations or index >= len(self.rotations):
            return
        i, j, _ = self.rotations[index]
        angle = float(value)
        self.rotations[index] = (i, j, angle)
        self.rotation_rows[index]['label'].configure(text=f'{angle:+.1f}°')
        self.schedule_render()

    def _add_rotation(self):
        n = len(self._axis_labels())
        # Prefer a distinct plane not already in use.
        existing = {(i, j) for i, j, _ in self.rotations}
        i, j = next(((i, j) for i in range(n) for j in range(i+1, n)
                     if (i, j) not in existing), (0, 1))
        self.rotations.append((i, j, 0.0))
        self._rebuild_rotation_rows()
        self.root.after_idle(lambda: self.rotation_canvas.yview_moveto(1.0))
        self.schedule_render()

    def _remove_rotation(self, index):
        if index >= len(self.rotations):
            return
        self.rotations.pop(index)
        self._rebuild_rotation_rows()
        self.schedule_render()

    def _reset_rotation_angles(self):
        self.rotations = [(i, j, 0.0) for i, j, _ in self.rotations]
        self._rebuild_rotation_rows()
        self.schedule_render()

    def _projection(self):
        names = self._axis_labels()
        p = 2 if self.view_var.get() == '2D' else 3
        axes = tuple(-1 if sv.get() == '0 (fixed)' else names.index(sv.get())
                     for sv in self.axis_vars[:p])
        return projection_matrix(len(names), axes, self.rotations), axes

    def _play_windows_error_sound(self):
        """Ask Windows to play the current system error alert (SystemHand).

        MessageBeep(MB_ICONHAND) honors the user's Windows sound scheme and
        accessibility settings. Tk's bell is the cross-platform fallback.
        """
        if sys.platform == 'win32':
            try:
                import ctypes
                if ctypes.windll.user32.MessageBeep(0x00000010):
                    return
            except (AttributeError, OSError):
                pass
        self.root.bell()

    def _show_input_error(self, field: str, detail: str, hint: str):
        """A distinct, modal input-validation dialog with correction advice."""
        self._play_windows_error_sound()
        popup = tk.Toplevel(self.root)
        popup.title(f'{field} error | Complex Cantor Explorer')
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.configure(background='#fffafa')

        content = tk.Frame(popup, background='#fffafa', padx=20, pady=17)
        content.pack(fill='both', expand=True)
        header = tk.Frame(content, background='#fffafa')
        header.pack(fill='x', pady=(0, 10))
        tk.Label(header, text='!', font=('Segoe UI', 20, 'bold'), foreground='white',
                 background='#b42318', width=2).pack(side='left', padx=(0, 11))
        tk.Label(header, text=f'{field}: input rejected', font=('Segoe UI', 13, 'bold'),
                 foreground='#912018', background='#fffafa').pack(side='left')
        tk.Label(content, text=detail, font=('Segoe UI', 10), justify='left',
                 foreground='#28344a', background='#fffafa', wraplength=450,
                 anchor='w').pack(fill='x', anchor='w', pady=(0, 13))
        tk.Label(content, text='How to correct it', font=('Segoe UI', 10, 'bold'),
                 foreground='#28344a', background='#fffafa').pack(anchor='w')
        tk.Label(content, text=hint, font=('Segoe UI', 10), justify='left',
                 foreground='#445368', background='#fffafa', wraplength=450,
                 anchor='w').pack(fill='x', anchor='w', pady=(4, 13))

        focus_target = {
            'Matrix': getattr(self, 'matrix_text', None),
            'Alphabet': getattr(self, 'alpha_text', None),
        }.get(field)

        def dismiss(_=None):
            popup.grab_release()
            popup.destroy()
            if focus_target is not None:
                focus_target.focus_set()

        ttk.Button(content, text='Correct input', command=dismiss).pack(anchor='e')
        popup.protocol('WM_DELETE_WINDOW', dismiss)
        popup.bind('<Escape>', dismiss)
        popup.bind('<Return>', dismiss)
        popup.update_idletasks()
        # Center over the desktop application's root window.
        width = max(420, popup.winfo_reqwidth())
        height = popup.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width()-width)//2
        y = self.root.winfo_rooty() + (self.root.winfo_height()-height)//2
        popup.geometry(f'{width}x{height}+{max(0,x)}+{max(0,y)}')
        popup.grab_set()
        popup.focus_set()

    @staticmethod
    def _bounded_integer(var: tk.Variable, field: str, lo: int, hi: int) -> int:
        try:
            value = int(var.get())
        except (ValueError, tk.TclError, TypeError) as exc:
            raise InputValidationError(field, f'Enter a whole number from {lo:,} to {hi:,}.',
                                       f'Correct the {field.lower()} field.') from exc
        if not lo <= value <= hi:
            raise InputValidationError(field, f'Value {value:,} is outside {lo:,} to {hi:,}.',
                                       f'Select a {field.lower()} in the allowed range.')
        return value

    @staticmethod
    def _check_invertible(M: np.ndarray):
        """Reject singular and numerically singular matrices before rendering."""
        d = M.shape[0]
        try:
            singular_values = np.linalg.svd(M, compute_uv=False)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise InputValidationError(
                'Matrix', 'Could not determine whether M is invertible.',
                'Check the matrix coefficients and try a well-conditioned matrix.') from exc
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        tolerance = d * np.finfo(np.float64).eps * largest
        if not np.isfinite(largest) or not np.isfinite(smallest) or smallest <= tolerance:
            raise InputValidationError(
                'Matrix',
                f'M is singular or numerically singular (smallest singular value: {smallest:.3g}). '
                'The expanding construction requires M^(-1), which cannot be computed reliably.',
                'Make the matrix full-rank (with linearly independent rows/columns). '
                'For an example, click "Load example for current d". '
                'A nearly singular matrix may require better-conditioned coefficients.')

    def _preflight(self):
        """Check every editor and parameter before numerical point generation."""
        d = self._bounded_integer(self.d_var, 'Dimension', 1, MAX_COMPLEX_DIM)
        k = self._bounded_integer(self.stage_var, 'Stage', 0, MAX_STAGE)
        limit = self._bounded_integer(self.max_points_var, 'Point budget', 1, 2_000_000)
        cells = self._bounded_integer(self.max_cells_var, 'Cell budget', 1, 10_000)
        # Keep projection controls consistent even when the user typed a
        # dimension and immediately pressed Generate without leaving the box.
        if d != self._valid_dimension:
            self._on_dimension_changed()
        try:
            M = parse_matrix(self._editor_contents(self.matrix_text), d)
        except ValueError as exc:
            raise InputValidationError(
                'Matrix', str(exc),
                f'For d={d}, enter exactly {d} rows of {d} complex coefficients. '
                'If you changed the dimension, your custom matrix was preserved; '
                'edit it or load the new default.') from exc
        self._check_invertible(M)
        try:
            A = parse_alphabet(self._editor_contents(self.alpha_text), d)
        except ValueError as exc:
            raise InputValidationError(
                'Alphabet', str(exc),
                f'For d={d}, each alphabet vector needs exactly {d} complex entries. '
                'Edit the custom alphabet or load the new default.') from exc
        return d, k, limit, cells, M, A

    def validate_inputs(self) -> bool:
        """Explicit validation without allocating or rendering any fractal."""
        try:
            self._preflight()
        except InputValidationError as exc:
            self._show_input_error(exc.field, str(exc), exc.hint)
            return False
        except (MemoryError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
            self._show_input_error('Validation', str(exc), 'Review the matrix and other inputs.')
            return False
        self.status_var.set('All inputs are valid. Click Generate / regenerate to update the fractal.')
        self.warning_var.set('')
        return True

    def generate(self):
        try:
            d, k, limit, cells, M, A = self._preflight()
            result = generate_points(M, A, k, max_points=limit)
            envelope = invariant_envelope(M, A)
            self.M, self.alphabet, self.result, self.envelope = M, A, result, envelope
            self._set_axis_choices(reset=False)
            self.render()
        except InputValidationError as exc:
            self._show_input_error(exc.field, str(exc), exc.hint)
        except (ValueError, np.linalg.LinAlgError, MemoryError, OverflowError, FloatingPointError) as exc:
            self._show_input_error('Generation', str(exc),
                                   'Check the matrix coefficients and stage/point limits, then try again.')

    def schedule_render(self):
        # Throttle, do not debounce: continuous dragging keeps refreshing even
        # when the pointer never pauses. Plotting reuses generated point data.
        if self._render_job is None:
            self._render_job = self.root.after(55, self.render)

    def render(self):
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = None
        self._update_camera_visibility()
        self._last_cell_bounds = None
        if self.result is None:
            return
        try:
            selected_dimension = self.d_var.get()
        except (ValueError, tk.TclError):
            return  # Dimension is currently being edited.
        if self.M is not None and self.M.shape[0] != selected_dimension:
            # A custom input was preserved on dimension change; no projection
            # can mix the old generated data with the newly selected axes.
            return
        try:
            P, axes = self._projection()
            points = self.result.points
            count = len(points)
            is3d = self.view_var.get() == '3D'
            mode = self.display_var.get()
            max_plot_points = 25_000 if is3d else 120_000
            n = min(count, max_plot_points)
            idx = np.linspace(0, count-1, num=n, dtype=np.int64) if n < count else slice(None)
            displayed = project(points[idx], P)
            drawn_cells = 0
            cell_bounds = None

            # Read the previous mouse camera state (matplotlib rotates on drag).
            if self.axes is not None and hasattr(self.axes, 'azim'):
                old_elev, old_azim = float(self.axes.elev), float(self.axes.azim)
                self.elev_var.set(old_elev)
                self.azim_var.set(old_azim)

            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d' if is3d else None)
            self.axes = ax
            ax.set_facecolor('#fcfdff')
            if mode in ('Cells', 'Both'):
                drawn_cells = self._render_cells(ax, P, is3d)
                if getattr(self, '_last_cell_bounds', None) is not None:
                    cell_bounds = self._last_cell_bounds

            if mode in ('Points', 'Both'):
                if is3d:
                    ax.scatter(displayed[:,0], displayed[:,1], displayed[:,2],
                               s=2.1 if n > 2000 else 8, c=BLUE, alpha=.85,
                               depthshade=False, linewidths=0, rasterized=False)
                else:
                    ax.scatter(displayed[:,0], displayed[:,1],
                               s=2.5 if n > 2000 else 10, c=BLUE, alpha=.85,
                               linewidths=0, rasterized=False)
            if mode == 'Cells' and cell_bounds is not None:
                lo, hi = cell_bounds
            else:
                lo, hi = displayed.min(axis=0), displayed.max(axis=0)
                if cell_bounds is not None:
                    lo = np.minimum(lo, cell_bounds[0])
                    hi = np.maximum(hi, cell_bounds[1])
            self._center = (lo + hi) / 2
            self._span = max(float(np.max(hi-lo))*1.12, 1e-9)
            if mode == 'Cells' and self.result.stage == 0:
                self._span = max(self._span, 0.1)
            labels = self._axis_labels()
            axis_label = []
            for a in axes:
                axis_label.append(('0' if a == -1 else labels[a]))
            ax.set_xlabel('Projection X  (' + axis_label[0] + ')', labelpad=10)
            ax.set_ylabel('Projection Y  (' + axis_label[1] + ')', labelpad=10)
            if is3d:
                ax.set_zlabel('Projection Z  (' + axis_label[2] + ')', labelpad=10)
                ax.set_box_aspect((1,1,1), zoom=.91)
                ax.view_init(elev=float(self.elev_var.get()), azim=float(self.azim_var.get()))
            else:
                ax.set_aspect('equal', adjustable='box')
                ax.grid(alpha=.20)
            ax.set_title(f'Stage k = {self.result.stage}   |   C^{self.M.shape[0]}   |   {"all words" if self.result.exact else "sampled words"}',
                         pad=19, color=INK, fontsize=12)
            self._apply_zoom(draw=False)
            self.figure.tight_layout(pad=2.2)
            self.canvas.draw_idle()

            coverage = 'exact' if self.result.exact else 'uniform word sample'
            self.status_var.set(f'{self.result.words:,} digit words; {count:,} stored ({coverage}). '
                                f'{n:,} plotted point markers; {drawn_cells:,} cell geometries. '
                                f'Maximum condition number estimate: {self.result.condition_number:.3g}.')
            warnings = []
            if not self.result.expanding:
                warnings.append('M is invertible but not strictly expanding; a compact attractor is not guaranteed.')
            if self.result.condition_number > 1e10:
                warnings.append('M is ill-conditioned: numerical accuracy may be poor.')
            if mode != 'Points':
                warnings.append(self.envelope.description)
                if drawn_cells < count:
                    warnings.append('Cells are drawn for only a subset of stored words.')
            if n < count and mode != 'Cells':
                warnings.append('The 3D/2D renderer has thinned the displayed point markers.')
            self.warning_var.set('  '.join(warnings))
        except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:
            self.warning_var.set(f'Visualization error: {exc}')

    def _render_cells(self, ax, P, is3d) -> int:
        count = len(self.result.points)
        budget = int(self.max_cells_var.get())
        budget = min(budget, 90) if is3d else min(budget, 1500)
        n = min(count, budget)
        idx = np.linspace(0, count-1, num=n, dtype=np.int64) if n < count else slice(None)
        centers = project(self.result.points[idx], P)
        shift, shape = projected_cell_geometry(self.M, self.result.stage, P, self.envelope)
        centers += shift
        verts = shape.vertices
        lo = centers.min(axis=0) + verts.min(axis=0)
        hi = centers.max(axis=0) + verts.max(axis=0)
        self._last_cell_bounds = (lo, hi)
        if shape.kind == 'polygon':
            poly = centers[:,None,:] + verts[None,:,:]
            if is3d:
                ax.add_collection3d(Poly3DCollection(poly, facecolor=ORANGE,
                         edgecolor='#c07c17', linewidth=.25, alpha=.24))
            else:
                ax.add_collection(PolyCollection(poly, facecolor=ORANGE,
                         edgecolor='#be821d', linewidth=.5, alpha=.29))
        elif shape.kind == 'triangles':
            triangles = verts[shape.faces]
            poly = (centers[:,None,None,:] + triangles[None,:,:,:]).reshape((-1, 3, 3))
            ax.add_collection3d(Poly3DCollection(poly, facecolor=ORANGE,
                 edgecolor='#cc9445', linewidth=.12, alpha=.23))
        elif shape.kind == 'segment':
            segments = centers[:,None,:] + verts[None,:,:]
            if is3d:
                ax.add_collection3d(Line3DCollection(segments, colors=ORANGE, linewidths=2.6, alpha=.9))
            else:
                ax.add_collection(LineCollection(segments, colors=ORANGE, linewidths=3.0, alpha=.9))
        else:
            if is3d:
                ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=ORANGE, s=20)
            else:
                ax.scatter(centers[:,0], centers[:,1], c=ORANGE, s=20)
        return n

    def _apply_zoom(self, _=None, *, draw=True):
        if self.axes is None or self._center is None:
            return
        zoom = max(.25, float(self.zoom_var.get()))
        radius = self._span/(2*zoom)
        cx, cy = self._center[:2]
        self.axes.set_xlim(cx-radius, cx+radius)
        self.axes.set_ylim(cy-radius, cy+radius)
        if len(self._center) == 3:
            cz = self._center[2]
            self.axes.set_zlim(cz-radius, cz+radius)
        if draw:
            self.canvas.draw_idle()

    def _mouse_zoom(self, event):
        if event.inaxes != self.axes:
            return
        factor = 1.22 if event.button == 'up' else (1/1.22 if event.button == 'down' else 1)
        self.zoom_var.set(min(12, max(.25, self.zoom_var.get() * factor)))
        self._apply_zoom()

    def _mouse_release(self, event):
        if self.axes is not None and hasattr(self.axes, 'azim') and event.inaxes == self.axes:
            self._syncing = True
            try:
                self.elev_var.set(float(self.axes.elev))
                self.azim_var.set(((float(self.axes.azim)+180)%360)-180)
            finally:
                self._syncing = False

    def _camera_changed(self):
        if self._syncing or self.axes is None or not hasattr(self.axes, 'view_init'):
            return
        self.axes.view_init(elev=float(self.elev_var.get()), azim=float(self.azim_var.get()))
        self.canvas.draw_idle()

    def export(self, kind: str):
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension='.'+kind,
            filetypes=[('PNG image','*.png')] if kind == 'png' else [('Scalable Vector Graphics','*.svg')],
            initialfile=f'cantor_d{self.M.shape[0]}_stage{self.result.stage}.{kind}')
        if not filename:
            return
        try:
            self.figure.savefig(filename, format=kind, dpi=220, facecolor='white')
            self.status_var.set('Exported displayed projection to ' + filename)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Export failed', str(exc))

    def _preset(self):
        return {
            'format': 'complex-cantor-explorer-v1',
            'dimension': self.d_var.get(),
            'stage': self.stage_var.get(),
            'point_budget': self.max_points_var.get(),
            'cell_budget': self.max_cells_var.get(),
            'matrix': self.matrix_text.get('1.0', 'end').strip(),
            'alphabet': self.alpha_text.get('1.0', 'end').strip(),
            'view': self.view_var.get(),
            'display': self.display_var.get(),
            'axes': [v.get() for v in self.axis_vars],
            'rotations': self.rotations,
            'elevation': self.elev_var.get(),
            'azimuth': self.azim_var.get(),
            'zoom': self.zoom_var.get(),
        }

    def save_preset(self):
        path = filedialog.asksaveasfilename(defaultextension='.json',
                   filetypes=[('Explorer preset', '*.json')], initialfile='cantor_preset.json')
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._preset(), indent=2), encoding='utf-8')
            self.status_var.set('Saved preset: ' + path)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Save failed', str(exc))

    def load_preset(self):
        path = filedialog.askopenfilename(filetypes=[('Explorer preset','*.json')])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if data.get('format') != 'complex-cantor-explorer-v1':
                raise ValueError('Unrecognized preset format.')
            d = int(data['dimension'])
            k = int(data['stage'])
            if not (1 <= d <= MAX_COMPLEX_DIM and 0 <= k <= MAX_STAGE):
                raise ValueError('Invalid dimension or iteration stage in preset.')
            # Validate content before mutating the interface.
            preset_matrix = parse_matrix(data['matrix'], d)
            self._check_invertible(preset_matrix)
            parse_alphabet(data['alphabet'], d)
            namecount = 2*d
            rotations = []
            for entry in data.get('rotations', []):
                i, j, angle = int(entry[0]), int(entry[1]), float(entry[2])
                if not (0 <= i < namecount and 0 <= j < namecount and i != j and np.isfinite(angle)):
                    raise ValueError('Invalid rotation in preset.')
                rotations.append((i,j,angle))
            if data.get('view') not in ('2D','3D') or data.get('display') not in ('Points','Cells','Both'):
                raise ValueError('Invalid rendering mode in preset.')
            self._setting_dim = True
            try:
                self.d_var.set(d)
                self._valid_dimension = d
            finally:
                self._setting_dim = False
            self._set_axis_choices(reset=True)
            self.stage_var.set(k)
            self.max_points_var.set(str(data['point_budget']))
            self.max_cells_var.set(str(data['cell_budget']))
            self._replace_editor(self.matrix_text, data['matrix'])
            self._replace_editor(self.alpha_text, data['alphabet'])
            self._default_matrix_text, self._default_alphabet_text = example_input(d)
            self.matrix_text.configure(height=max(3, min(7, d + 1)))
            self.view_var.set(data['view'])
            self.display_var.set(data['display'])
            names = self._axis_labels() + ['0 (fixed)']
            for v, value in zip(self.axis_vars, data.get('axes', [])):
                if value in names:
                    v.set(value)
            self.rotations = rotations
            self._rebuild_rotation_rows()
            self.elev_var.set(float(data.get('elevation',24)))
            self.azim_var.set(float(data.get('azimuth',-60)))
            self.zoom_var.set(float(data.get('zoom',1)))
            self.generate()
        except InputValidationError as exc:
            self._show_input_error(exc.field, str(exc), exc.hint)
        except (OSError, KeyError, ValueError, TypeError, OverflowError, json.JSONDecodeError) as exc:
            self._show_input_error('Preset', str(exc),
                                   'Check the JSON preset or choose a different file.')


def main():
    root = tk.Tk()
    CantorExplorer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
