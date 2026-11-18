import itertools
import numpy as np
import pytest

from cantor_core import (
    parse_matrix, parse_alphabet, generate_points, to_real, real_matrix,
    invariant_envelope, compose_rotations, projection_matrix, project,
    projected_cell_geometry,
)


def sort_complex(vals):
    return sorted([complex(v) for v in np.asarray(vals).ravel()], key=lambda z:(z.real,z.imag))


def test_standard_middle_thirds_stage2():
    out = generate_points(np.array([[3]]), np.array([[0],[2]]), 2)
    assert out.exact
    assert out.words == 4
    assert np.allclose(sort_complex(out.points), [0,2/9,2/3,8/9])


def test_stage_zero():
    out = generate_points(np.eye(3)*3, np.array([[0,0,0],[2,0,0]]), 0)
    assert out.words == 1
    assert out.points.shape == (1,3)
    assert np.array_equal(out.points, np.zeros((1,3)))


def test_complex_matrix_matches_explicit_formula():
    M = np.array([[3+0.4j, .4-.3j],[0.0, 2.8+.2j]])
    digits = np.array([[0,0],[2+1j,0],[0,1.8-0.7j]])
    B = np.linalg.inv(M)
    k = 4
    expected = []
    powers = [np.linalg.matrix_power(B,j) for j in range(1,k+1)]
    for seq in itertools.product(digits, repeat=k):
        expected.append(sum((powers[j] @ seq[j] for j in range(k)), np.zeros(2,complex)))
    observed = generate_points(M,digits,k,max_points=1000)
    sorter = lambda rows: sorted(tuple(np.round(np.concatenate((v.real,v.imag)),11)) for v in rows)
    assert observed.exact
    assert sorter(observed.points) == sorter(expected)


def test_complex_parser_and_reject_arbitrary_python():
    m = parse_matrix('3 + 1j, 0\n 0, 3-1j', 2)
    a = parse_alphabet('0, 0\n2+3j, -1j',2)
    assert m[0,0] == 3+1j
    assert a[1,1] == -1j
    with pytest.raises(ValueError):
        parse_matrix("__import__('os').system('whoami')",1)
    with pytest.raises(ValueError):
        parse_alphabet('NaN', 1)
    with pytest.raises(ValueError):
        parse_matrix('3', 2)


def test_invalid_singular_matrix():
    with pytest.raises(ValueError):
        generate_points(np.array([[1,2],[2,4]]), np.array([[0,0]]), 2)


def test_sample_mode_deterministic():
    M = np.array([[3]])
    A = np.array([[0], [2], [1+1j]])
    one = generate_points(M, A, 8, max_points=1234, seed=42)
    two = generate_points(M, A, 8, max_points=1234, seed=42)
    assert one.words == 3**8
    assert not one.exact
    assert one.points.shape == (1234,1)
    assert np.array_equal(one.points, two.points)


def test_projection_rotation_and_realification():
    z = np.array([[1+2j, 3-4j]])
    assert np.array_equal(to_real(z),[[1,3,2,-4]])
    B = np.array([[1+1j]])
    assert np.array_equal(real_matrix(B),[[1,-1],[1,1]])
    rot = compose_rotations(4, [(0,2,90)])
    assert np.allclose(rot.T@rot,np.eye(4),atol=1e-14)
    p = projection_matrix(4,(0,2,1),[(0,2,90)])
    assert np.allclose(project(z,p), [[-2,1,3]])


def test_invariant_box_contains_middle_thirds():
    M = np.array([[3]],complex)
    A = np.array([[0],[2]],complex)
    q = invariant_envelope(M,A)
    assert q.rigorous
    assert np.allclose(q.center, [0.5,0])
    assert np.allclose(q.radii, [0.5,0])
    p = projection_matrix(2,(0,1),[])
    shift, shape = projected_cell_geometry(M,1,p,q)
    assert shape.kind == 'segment'
    assert np.allclose(shift,[1/6,0])
    # First cylinder is [0,1/3], second [2/3,1].
    assert np.allclose(np.sort(shape.vertices[:,0] + shift[0]),[0,1/3])


def test_nonexpanding_matrix_has_reference_cells():
    M = np.eye(2)
    q = invariant_envelope(M,np.array([[0,0],[2,2]],complex))
    assert not q.rigorous


def test_full_3d_zonotope_and_rank_two():
    M = np.eye(2,dtype=complex)*3
    A = np.array([[0,0],[2,0],[0,2],[2,2]],complex)
    q = invariant_envelope(M,A)
    p = projection_matrix(4,(0,1,2),[])
    _, shape = projected_cell_geometry(M,2,p,q)
    assert shape.kind in ('polygon','triangles')
    # Need actual full three-dimensional independent digit differences.
    A3 = np.array([[0,0],[2,0],[0,2],[2j,0]],complex)
    q3 = invariant_envelope(M,A3)
    _, shape3 = projected_cell_geometry(M,1,p,q3)
    assert shape3.kind == 'triangles'
    assert shape3.faces.shape[1] == 3


def test_anisotropic_nonnormal_expanding_bounds():
    M = np.array([[2.,100.],[0.,2.]])
    A = np.array([[0,0],[1,1]],complex)
    q = invariant_envelope(M,A)
    assert q.rigorous
    assert np.isfinite(q.radii).all()
    # Coordinatewise invariance check (for either conservative or tight Q).
    T = real_matrix(np.linalg.inv(M))
    center = q.center
    radius = q.radii
    for a in to_real(A):
        b = T@(center+a)
        assert np.all(np.abs(b-center) + np.abs(T)@radius <= radius+1e-6*max(1,np.max(radius)))
