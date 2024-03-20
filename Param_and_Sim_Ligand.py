import os
import numpy as np

from openmm.app import *
from openmm import *
from openmm.unit import *

#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange

name = '1-Butamine_large'
smiles = 'CCCCN'

def write_structure(sim: Simulation, pdb_fn: str):
    with open(pdb_fn, 'w') as f:
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
    print(f'Wrote: {pdb_fn}')

mol = openff.toolkit.Molecule.from_smiles(smiles)
mol.generate_conformers()
ff = openff.toolkit.ForceField('openff-2.1.0.offxml')

cubic_box = openff.units.Quantity(30 * np.eye(3), openff.units.unit.angstrom)
mol.conformers[0] += 15*openff.units.unit.angstrom

interchange = openff.interchange.Interchange.from_smirnoff(topology=[mol], force_field=ff, box=cubic_box)

interchange.to_prmtop(f'{name}.prmtop')

positions, system, topology = interchange.positions.magnitude, interchange.to_openmm_system(), interchange.to_openmm_topology()

print(positions, system, topology)

#Lets restrain the first atom to it's intial position
rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x0,y0,z0)^2')
rest.addGlobalParameter('fc_pos', 200)
rest.addPerParticleParameter('x0')
rest.addPerParticleParameter('y0')
rest.addPerParticleParameter('z0')
rest.addParticle(0, positions[0])
system.addForce(rest)

n_total, n_dcd, n_std, timestep, temp = 5000000, 500, 500, 2.0*femtosecond, 300*kelvin

simulation = Simulation(topology, system, LangevinIntegrator(temp, 1/picosecond, timestep))
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temp)

SDR = StateDataReporter(f'{name}.stdout', n_std, step=True, time=True,
                        potentialEnergy=True, temperature=True, remainingTime=True,
                        totalSteps=n_total, separator='     ')
simulation.reporters.append(SDR)
#Trajectory Reporter (You need this to keep your coordinates!)
DCR = DCDReporter(f'{name}.dcd', n_dcd)
simulation.reporters.append(DCR)

simulation.step(n_total)

write_structure(simulation, f'{name}.pdb')
