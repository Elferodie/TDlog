from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
import numpy as np


class Simulationn:
    def __init__(self, file_psf, file_pdb, number_atoms):
        self.number_atoms = number_atoms
        psf = CharmmPsfFile(file_psf)
        pdb = PDBFile(file_pdb)
        params = CharmmParameterSet('Fichiers_alanine-dipeptide/top_all27_prot_lipid.rtf', 'Fichiers_alanine-dipeptide/par_all27_prot_lipid.prm')

        # System Configuration
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.4 * nanometers,
                                  constraints=HBonds)

        # Integration Options
        dt = 0.002 * picoseconds
        temperature = 300.00 * kelvin
        friction = 2 / (pico * second)
        integrator = LangevinIntegrator(temperature, friction, dt)
        integrator.setConstraintTolerance(0.00001)

        # do minimization, perform equilibration, then save the state of the simulation
        self.equilibrationSteps = 25000
        platform = Platform.getPlatformByName('CPU')
        platformProperties = {'CpuThreads': '1'}

        self.simulation = Simulation(psf.topology, system, integrator, platform, platformProperties)
        self.simulation.context.setPositions(pdb.positions)

        # Minimize and Equilibrate
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(temperature)

    def coordinates(self, name_file):
        array = np.zeros(self.number_atoms*3)
        file = open(name_file, 'r')
        for i in range(7):
            file.readline()
        potential = file.readline().split('"')[1]
        file.readline()
        for i in range(22):
            line = file.readline()
            line_split = line.split('"')
            array[3 * i] = float(line_split[1])
            array[3 * i + 1] = float(line_split[3])
            array[3 * i + 2] = float(line_split[5])
        return array, potential

    def trajectory(self):
        traj = np.empty([self.equilibrationSteps, self.number_atoms*3])
        pot = np.empty(self.equilibrationSteps)

        self.simulation.currentStep = 0
        self.simulation.context.setTime(0.0)

        for step in range(self.equilibrationSteps):
            self.simulation.step(1)
            state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            f = open('State.xml', 'w')
            f.write(XmlSerializer.serialize(state))
            f.close()
            traj[step], pot[step] = self.coordinates('State.xml')
        return traj, pot
