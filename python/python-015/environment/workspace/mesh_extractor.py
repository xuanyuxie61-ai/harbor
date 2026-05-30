
import numpy as np
from typing import Tuple, Optional


def mesh2d_extract(nodeco: np.ndarray, elnode: np.ndarray,
                    bdynde: Optional[np.ndarray] = None) -> dict:
    node_num = nodeco.shape[0]
    element_num = elnode.shape[0]
    
    mesh = {
        'dim': 2,
        'node_num': node_num,
        'element_num': element_num,
        'nodeco': nodeco,
        'elnode': elnode,
        'bdynde': bdynde,
        'element_order': 3
    }
    
    return mesh


def mesh3d_extract(nodeco: np.ndarray, elnode: np.ndarray,
                    bdynde: Optional[np.ndarray] = None) -> dict:
    node_num = nodeco.shape[0]
    element_num = elnode.shape[0]
    element_order = elnode.shape[1]
    
    mesh = {
        'dim': 3,
        'node_num': node_num,
        'element_num': element_num,
        'nodeco': nodeco,
        'elnode': elnode,
        'bdynde': bdynde,
        'element_order': element_order
    }
    
    return mesh


def node_values_to_elements(node_values: np.ndarray, elnode: np.ndarray,
                            element_order: int = 3) -> np.ndarray:
    element_num = elnode.shape[0]
    node_values = np.asarray(node_values)
    
    if node_values.ndim == 1:
        element_values = np.zeros(element_num)
        for i in range(element_num):
            element_values[i] = np.mean(node_values[elnode[i]])
    else:
        D = node_values.shape[1]
        element_values = np.zeros((element_num, D))
        for i in range(element_num):
            element_values[i] = np.mean(node_values[elnode[i]], axis=0)
    
    return element_values


def mesh_to_xml_string(mesh: dict) -> str:
    dim = mesh['dim']
    node_num = mesh['node_num']
    element_num = mesh['element_num']
    nodeco = mesh['nodeco']
    elnode = mesh['elnode']
    element_order = mesh.get('element_order', 3)
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append('<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">')
    
    if dim == 2:
        lines.append('  <mesh celltype="triangle" dim="2">')
    else:
        lines.append(f'  <mesh celltype="tetrahedron" dim="{dim}">')
    
    lines.append(f'    <vertices size="{node_num}">')
    for node in range(node_num):
        if dim == 2:
            lines.append(f'      <vertex index="{node}" x="{nodeco[node, 0]:.10g}" y="{nodeco[node, 1]:.10g}"/>')
        else:
            lines.append(f'      <vertex index="{node}" x="{nodeco[node, 0]:.10g}" y="{nodeco[node, 1]:.10g}" z="{nodeco[node, 2]:.10g}"/>')
    lines.append('    </vertices>')
    
    lines.append(f'    <cells size="{element_num}">')
    for elem in range(element_num):
        if dim == 2 and element_order == 3:
            lines.append(f'      <triangle index="{elem}" v0="{elnode[elem, 0]}" v1="{elnode[elem, 1]}" v2="{elnode[elem, 2]}"/>')
        elif dim == 3 and element_order == 4:
            lines.append(f'      <tetrahedron index="{elem}" v0="{elnode[elem, 0]}" v1="{elnode[elem, 1]}" v2="{elnode[elem, 2]}" v3="{elnode[elem, 3]}"/>')
        else:

            v_str = ' '.join([f'v{i}="{elnode[elem, i]}"' for i in range(element_order)])
            lines.append(f'      <cell index="{elem}" {v_str}/>')
    lines.append('    </cells>')
    lines.append('  </mesh>')
    lines.append('</dolfin>')
    
    return '\n'.join(lines)


def create_fermi_surface_mesh(k_points: np.ndarray, energies: np.ndarray,
                               e_fermi: float, tolerance: float = 0.1) -> dict:
    mask = np.abs(energies - e_fermi) < tolerance
    fs_points = k_points[mask, :2]
    
    if len(fs_points) < 3:
        return {
            'dim': 2,
            'node_num': len(fs_points),
            'element_num': 0,
            'nodeco': fs_points,
            'elnode': np.zeros((0, 3), dtype=int),
            'bdynde': None,
            'element_order': 3
        }
    
    from triangulation_mesh import delaunay_triangulate_2d
    triangles = delaunay_triangulate_2d(fs_points)
    
    mesh = {
        'dim': 2,
        'node_num': len(fs_points),
        'element_num': len(triangles),
        'nodeco': fs_points,
        'elnode': triangles,
        'bdynde': None,
        'element_order': 3
    }
    
    return mesh
