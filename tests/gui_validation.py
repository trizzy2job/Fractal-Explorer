"""Regression tests for editor preservation and GUI preflight validation.

Run under Windows or use xvfb-run -a python tests/gui_validation.py on Linux.
"""
import sys
from pathlib import Path
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cantor_core import parse_alphabet, parse_matrix, MAX_STAGE
from cantor_explorer import CantorExplorer, example_input


def contents(editor):
    return editor.get('1.0', 'end-1c').strip()


def run():
    root = tk.Tk()
    app = CantorExplorer(root)
    root.update()
    failures = []

    # Both untouched examples follow d.
    app.d_var.set(3)
    app._on_dimension_changed()
    assert (contents(app.matrix_text), contents(app.alpha_text)) == example_input(3)
    assert parse_matrix(contents(app.matrix_text), 3).shape == (3, 3)

    # Editing only the matrix keeps it, but the untouched alphabet follows d.
    custom_matrix = '4, 1, 0\n0, 4, 0\n0, 0, 4'
    app._replace_editor(app.matrix_text, custom_matrix)
    app.d_var.set(4)
    app._on_dimension_changed()
    assert contents(app.matrix_text) == custom_matrix
    assert contents(app.alpha_text) == example_input(4)[1]

    # Editing both makes both remain unchanged across changes of d and k.
    custom_alphabet = '0, 0, 0, 0\n1, 2, 3, 4'
    app._replace_editor(app.alpha_text, custom_alphabet)
    app.stage_var.set(7)
    assert contents(app.matrix_text) == custom_matrix
    assert contents(app.alpha_text) == custom_alphabet
    app.d_var.set(5)
    app._on_dimension_changed()
    assert contents(app.matrix_text) == custom_matrix
    assert contents(app.alpha_text) == custom_alphabet

    # An explicit "Load example" deliberately restores both defaults.
    app._load_example()
    assert (contents(app.matrix_text), contents(app.alpha_text)) == example_input(5)
    # Editing only the alphabet means only the matrix follows the new dimension.
    app._replace_editor(app.alpha_text, custom_alphabet)
    app.d_var.set(2)
    app._on_dimension_changed()
    assert contents(app.matrix_text) == example_input(2)[0]
    assert contents(app.alpha_text) == custom_alphabet

    # Intercept modal dialogs to assert correct failure reporting non-interactively.
    app._show_input_error = lambda field, detail, hint: failures.append((field, detail, hint))
    original_result = app.result
    assert not app.validate_inputs()
    assert failures[-1][0] == 'Alphabet'
    assert 'expected 2 entries' in failures[-1][1]
    app.generate()
    assert app.result is original_result, 'Failure must preserve the last valid result.'

    # A singular M is rejected before entering the numerical generator.
    app._replace_editor(app.alpha_text, '0, 0\n2, 0')
    app._replace_editor(app.matrix_text, '1, 2\n2, 4')
    assert not app.validate_inputs()
    assert failures[-1][0] == 'Matrix'
    assert 'singular' in failures[-1][1].lower()
    app.generate()
    assert app.result is original_result

    # Bad stage limits also produce a tailored validation error.
    app.stage_var.set(MAX_STAGE + 1)
    assert not app.validate_inputs()
    assert failures[-1][0] == 'Stage'
    app.stage_var.set(3)

    # Check the real error dialog separately, including its alert sound call.
    played = []
    app._play_windows_error_sound = lambda: played.append(True)
    # We only create a real popup at this step, then dismiss it programmatically.
    app._show_input_error = CantorExplorer._show_input_error.__get__(app)
    app._show_input_error('Matrix', 'M is singular.', 'Use an invertible matrix.')
    root.update()
    dialogs = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(dialogs) == 1
    assert 'Matrix error' in dialogs[0].title()
    assert played == [True]
    dialogs[0].event_generate('<Escape>')
    root.update()
    assert not [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    root.destroy()
    print('GUI validation tests passed (defaults, preservation, stage, singularity, popup, alert).')


if __name__ == '__main__':
    run()
