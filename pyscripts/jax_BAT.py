from MDAnalysis.analysis.bat import BAT 
import logging
import jax
import jax.numpy as jnp
import copy
import timeit 

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase

from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from MDAnalysis.lib.mdamath import make_whole

from MDAnalysis.due import due, Doi

logger = logging.getLogger(__name__)

def _sort_atoms_by_mass(atoms, reverse=False):
    r"""Sorts a list of atoms by name and then by index

    The atom index is used as a tiebreaker so that the ordering is reproducible.

    Parameters
    ----------
    ag_o : list of Atoms
        List to sort
    reverse : bool
        Atoms will be in descending order

    Returns
    -------
    ag_n : list of Atoms
        Sorted list
    """
    return sorted(atoms, key=lambda a: (a.mass, a.index), reverse=reverse)


def _find_torsions(root, atoms):
    """Constructs a list of torsion angles

    Parameters
    ----------
    root : AtomGroup
        First three atoms in the coordinate system
    atoms : AtomGroup
        Atoms that are allowed to be part of the torsion angle

    Returns
    -------
    torsions : list of AtomGroup
        list of AtomGroup objects that define torsion angles
    """
    start = timeit.default_timer()
    print("Starting torsions calculation.")
    torsions = []
    selected_atoms = list(root)
    while len(selected_atoms) < len(atoms):
        torsionAdded = False
        for a1 in selected_atoms:
            # Find a0, which is a new atom connected to the selected atom
            a0_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                if (a in atoms) and (a not in selected_atoms))
            for a0 in a0_list:
                # Find a2, which is connected to a1, is not a terminal atom,
                # and has been selected
                a2_list = _sort_atoms_by_mass(a for a in a1.bonded_atoms \
                    if (a!=a0) and len(a.bonded_atoms)>1 and \
                        (a in atoms) and (a in selected_atoms))
                for a2 in a2_list:
                    # Find a3, which is
                    # connected to a2, has been selected, and is not a1
                    a3_list = _sort_atoms_by_mass(a for a in a2.bonded_atoms \
                        if (a!=a1) and \
                            (a in atoms) and (a in selected_atoms))
                    for a3 in a3_list:
                        # Add the torsion to the list of torsions
                        torsions.append(mda.AtomGroup([a0, a1, a2, a3]))
                        # Add the new atom to selected_atoms
                        # which extends the loop
                        selected_atoms.append(a0)
                        torsionAdded = True
                        break  # out of the a3 loop
                    break  # out of the a2 loop
        if torsionAdded is False:
            print('Selected atoms:')
            print([a.index + 1 for a in selected_atoms])
            print('Torsions found:')
            print([list(t.indices + 1) for t in torsions])
            raise ValueError('Additional torsions not found.')
    stop = timeit.default_timer()
    print(f"Torsions calculation took {stop-start}s.")
    return torsions

class BAT_jax(BAT):

    @due.dcite(Doi("10.1002/jcc.26036"),
               description="Bond-Angle-Torsions Coordinate Transformation",
               path="MDAnalysis.analysis.bat.BAT")
    def __init__(self, ag, initial_atom=None, filename=None, **kwargs):
        r"""Parameters
        ----------
        ag : AtomGroup or Universe
            Group of atoms for which the BAT coordinates are calculated.
            `ag` must have a bonds attribute.
            If unavailable, bonds may be guessed using
            :meth:`AtomGroup.guess_bonds <MDAnalysis.core.groups.AtomGroup.guess_bonds>`.
            `ag` must only include one molecule.
            If a trajectory is associated with the atoms, then the computation
            iterates over the trajectory.
        initial_atom : :class:`Atom <MDAnalysis.core.groups.Atom>`
            The atom whose Cartesian coordinates define the translation
            of the molecule. If not specified, the heaviest terminal atom
            will be selected.
        filename : str
            Name of a numpy binary file containing a saved bat array.
            If filename is not ``None``, the data will be loaded from this file
            instead of being recalculated using the run() method.

        Raises
        ------
        AttributeError
            If `ag` does not contain a bonds attribute
        ValueError
            If `ag` contains more than one molecule

        """
        super(BAT, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag

        # Check that the ag contains bonds
        if not hasattr(self._ag, 'bonds'):
            raise AttributeError('AtomGroup has no attribute bonds')
        if len(self._ag.fragments) > 1:
            raise ValueError('AtomGroup has more than one molecule')

        # Determine the root
        # The initial atom must be a terminal atom
        terminal_atoms = _sort_atoms_by_mass(\
            [a for a in self._ag.atoms if len(a.bonds)==1], reverse=True)
        if (initial_atom is None):
            # Select the heaviest root atoms from the heaviest terminal atom
            initial_atom = terminal_atoms[0]
        elif (not initial_atom in terminal_atoms):
            raise ValueError('Initial atom is not a terminal atom')
        # The next atom in the root is bonded to the initial atom
        # Since the initial atom is a terminal atom, there is only
        # one bonded atom
        second_atom = initial_atom.bonded_atoms[0]
        # The last atom in the root is the heaviest atom
        # bonded to the second atom
        # If there are more than three atoms,
        # then the last atom cannot be a terminal atom.
        if self._ag.n_atoms != 3:
            third_atom = _sort_atoms_by_mass(\
                [a for a in second_atom.bonded_atoms \
                if (a in self._ag) and (a!=initial_atom) \
                and (a not in terminal_atoms)], \
                reverse=True)[0]
        else:
            third_atom = _sort_atoms_by_mass(\
                [a for a in second_atom.bonded_atoms \
                if (a in self._ag) and (a!=initial_atom)], \
                reverse=True)[0]
        self._root = mda.AtomGroup([initial_atom, second_atom, third_atom])

        # Construct a list of torsion angles
        self._torsions = _find_torsions(self._root, self._ag)

        # Get indices of the root and torsion atoms
        # in a Cartesian positions array that matches the AtomGroup
        self._root_XYZ_inds = [(self._ag.indices==a.index).nonzero()[0][0] \
            for a in self._root]
        self._torsion_XYZ_inds = [[(self._ag.indices==a.index).nonzero()[0][0] \
            for a in t] for t in self._torsions]

        # The primary torsion is the first torsion on the list
        # with the same central atoms
        prior_atoms = [sorted([a1, a2]) for (a0, a1, a2, a3) in self._torsions]
        self._primary_torsion_indices = [prior_atoms.index(prior_atoms[n]) \
            for n in range(len(prior_atoms))]
        self._unique_primary_torsion_indices = \
            list(set(self._primary_torsion_indices))

        self._ag1 = mda.AtomGroup([ag[0] for ag in self._torsions])
        self._ag2 = mda.AtomGroup([ag[1] for ag in self._torsions])
        self._ag3 = mda.AtomGroup([ag[2] for ag in self._torsions])
        self._ag4 = mda.AtomGroup([ag[3] for ag in self._torsions])

        if filename is not None:
            self.load(filename)

        self.VCartesian = jax.vmap(self.Cartesian)
    
    def _prepare(self):
        jax.dtypes.canonicalize_dtype(jnp.float64)
        self.results.bat = jnp.zeros(
                (self.n_frames, 3*self._ag.n_atoms), dtype=jnp.float64)

    def _single_frame(self):
    # Calculate coordinates based on the root atoms
    # The rotation axis is a normalized vector pointing from atom 0 to 1
    # It is described in two degrees of freedom
    # by the polar angle and azimuth
        if self._root.dimensions is None:
            (p0, p1, p2) = self._root.positions
        else:
            (p0, p1, p2) = make_whole(self._root, inplace=False)
        v01 = p1 - p0
        v21 = p1 - p2
        # Internal coordinates
        r01 = jnp.sqrt(jnp.dot(v01, v01))  
        # Distance between first two root atoms
        r12 = jnp.sqrt(jnp.dot(v21, v21))  
        # Distance between second two root atoms
        # Angle between root atoms
        a012 = jnp.arccos(jnp.maximum(-1., jnp.minimum(1., jnp.dot(v01, v21) /
                                (jnp.sqrt(jnp.dot(v01, v01)) * jnp.sqrt(jnp.dot(v21, v21))))))

        # External coordinates
        e = v01 / r01
        phi = jnp.arctan2(e[1], e[0])  # Polar angle
        theta = jnp.arccos(e[2])  # Azimuthal angle
        # Rotation to the z axis
        cp = jnp.cos(phi)
        sp = jnp.sin(phi)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        Rz = jnp.array([[cp * ct, ct * sp, -st], [-sp, cp, 0],
                        [cp * st, sp * st, ct]])
        pos2 = jnp.dot(Rz, p2 - p1)
        # Angle about the rotation axis
        omega = jnp.arctan2(pos2[1], pos2[0])
        root_based = jnp.concatenate((p0, jnp.array([phi, theta, omega, r01, r12, a012])))

        # Calculate internal coordinates from the torsion list
        bonds = calc_bonds(self._ag1.positions,self._ag2.positions,box=self._ag1.dimensions)
        angles = calc_angles(self._ag1.positions,self._ag2.positions,self._ag3.positions,box=self._ag1.dimensions)
        torsions = calc_dihedrals(self._ag1.positions,self._ag2.positions,self._ag3.positions,self._ag4.positions,box=self._ag1.dimensions)
        
        # When appropriate, calculate improper torsions
        shift = torsions[self._primary_torsion_indices]
        shift[self._unique_primary_torsion_indices] = 0.
        torsions -= shift
        
        # Wrap torsions to between -jnp.pi and jnp.pi
        torsions = ((torsions + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        self.results.bat = self.results.bat.at[self._frame_index, :].set(jnp.concatenate(
                (root_based, bonds, angles, torsions)))

    def save(self, filename):
        """Saves the bat trajectory in a file in numpy binary format

        See Also
        --------
        load: Loads the bat trajectory from a file in numpy binary format
        """
        jnp.save(filename, self.results.bat)

    def Cartesian(self, bat_frame):
        """Conversion of a single frame from BAT to Cartesian coordinates

        One application of this function is to determine the new
        Cartesian coordinates after modifying a specific torsion angle.

        Parameters
        ----------
        bat_frame : numpy.ndarray
            an array with dimensions (3N,) with external then internal
            degrees of freedom based on the root atoms, followed by the bond,
            angle, and (proper and improper) torsion coordinates.

        Returns
        -------
        XYZ : numpy.ndarray
            an array with dimensions (N,3) with Cartesian coordinates. The first
            dimension has the same ordering as the AtomGroup used to initialize
            the class. The molecule will be whole opposed to wrapped around a
            periodic boundary.
        """
        start = timeit.default_timer()
        print("Starting Cartesian calculation.")
   
        origin = bat_frame[:3]
        (phi, theta, omega) = bat_frame[3:6]
        (r01, r12, a012) = bat_frame[6:9]
        n_torsions = (self._ag.n_atoms - 3)
        bonds = bat_frame[9:n_torsions + 9]
        angles = bat_frame[n_torsions + 9:2 * n_torsions + 9]
        torsions = copy.deepcopy(bat_frame[2 * n_torsions + 9:])
        # When appropriate, convert improper to proper torsions
        shift = torsions[jnp.array(self._primary_torsion_indices)]
        shift = shift.at[jnp.array(self._unique_primary_torsion_indices)].set(0.)
        torsions += shift
        # Wrap torsions to between -np.pi and np.pi
        torsions = ((torsions + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        stop = timeit.default_timer()
        print(f"Midpoint 1 took {stop-start}s.")

        p0 = jnp.array([0., 0., 0.])
        p1 = jnp.array([0., 0., r01])
        p2 = jnp.array([r12 * jnp.sin(a012), 0., r01 - r12 * jnp.cos(a012)])

        co = jnp.cos(omega)
        so = jnp.sin(omega)
        Romega = jnp.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
        p2 = jnp.dot(Romega, p2)

        cp = jnp.cos(phi)
        sp = jnp.sin(phi)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        Re = jnp.array([[cp * ct, -sp, cp * st], [ct * sp, cp, sp * st], [-st, 0, ct]])
        p1 = jnp.dot(Re, p1)
        p2 = jnp.dot(Re, p2)

        p0 += origin
        p1 += origin
        p2 += origin

        XYZ = jnp.zeros((self._ag.n_atoms, 3))

        XYZ = XYZ.at[self._root_XYZ_inds[0]].set(p0)
        XYZ = XYZ.at[self._root_XYZ_inds[1]].set(p1)
        XYZ = XYZ.at[self._root_XYZ_inds[2]].set(p2)

        stop = timeit.default_timer()
        print(f"Midpoint 2 took {stop-start}s.")

        for ((a0, a1, a2, a3), r01, angle, torsion) in zip(self._torsion_XYZ_inds, bonds, angles, torsions):
            p1 = XYZ[a1]
            p3 = XYZ[a3]
            p2 = XYZ[a2]

            sn_ang = jnp.sin(angle)
            cs_ang = jnp.cos(angle)
            sn_tor = jnp.sin(torsion)
            cs_tor = jnp.cos(torsion)

            v21 = p1 - p2
            v21 /= jnp.sqrt(jnp.einsum('i,i->', v21, v21))
            v32 = p2 - p3
            v32 /= jnp.sqrt(jnp.einsum('i,i->', v32, v32))

            vp = jnp.cross(v32, v21)
            cs = jnp.einsum('i,i->', v21, v32)

            sn = jnp.maximum(jnp.sqrt(1.0 - cs * cs), 0.0000000001)
            vp = vp / sn
            vu = jnp.cross(vp, v21)

            XYZ = XYZ.at[a0].set(p1 + r01*(vu*sn_ang*cs_tor + vp*sn_ang*sn_tor - v21*cs_ang))
        
        stop = timeit.default_timer()
        print(f"Cartesian calculation took {stop-start}s.")
        
        return XYZ   