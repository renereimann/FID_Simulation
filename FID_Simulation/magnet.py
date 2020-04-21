from .units import *

class RingMagnet(object):
    """Class representing the magnetic field of a RingMagnet.

    The main magnetic field B0 is directing in y-direction.
    Deviations are described by multipoles.
    The strength of different multipoles can be set by
        magnet.An[multipole id] =  strength
    The magnetic field 3D vector can be calculated at any point (x,y,z) by
    calling the class instance.
    Further helper functions for pretty prints and unit handling are provided.
    """
    def __init__(self, B0):
        """
        Parameters:
        * B0: float, strength of main direction of the magnetic field.

        Note:
            * Main magnetic field is a dipole field in y-direction
            * You can set further multipole strength by
                  magnet.An[multipole id] = strength
            * Provide the correct unit for gradient strength, e.g. T/cm, ...
        """
        self.An = { # dipoles
                   1: 0*T,
                   2: B0,
                   3: 0*T,
                   # quadrupoles
                   4: 0*T/mm,
                   5: 0*T/mm,
                   6: 0*T/mm,
                   7: 0*T/mm,
                   8: 0*T/mm,
                   # sextupoles
                   9: 0*T/mm**2,
                  10: 0*T/mm**2,
                  11: 0*T/mm**2,
                  12: 0*T/mm**2,
                  13: 0*T/mm**2,
                  14: 0*T/mm**2,
                  15: 0*T/mm**2,
                  # octupole
                  16: 0*T/mm**3,
                  17: 0*T/mm**3,
                  18: 0*T/mm**3,
                  19: 0*T/mm**3,
                  20: 0*T/mm**3,
                  21: 0*T/mm**3,
                  22: 0*T/mm**3,
                  23: 0*T/mm**3,
                  24: 0*T/mm**3,
                 }

        self.P = { # dipoles
                   1: {"x": lambda x, y, z: 1, "y": lambda x, y, z: 0, "z": lambda x, y, z: 0},
                   2: {"x": lambda x, y, z: 0, "y": lambda x, y, z: 1, "z": lambda x, y, z: 0},
                   3: {"x": lambda x, y, z: 1, "y": lambda x, y, z: 0, "z": lambda x, y, z: 1},
                   # quadrupoles
                   4: {"x": lambda x, y, z: x, "y": lambda x, y, z: -y, "z": lambda x, y, z: 0},
                   5: {"x": lambda x, y, z: z, "y": lambda x, y, z: 0, "z": lambda x, y, z: x},
                   6: {"x": lambda x, y, z: 0, "y": lambda x, y, z: -y, "z": lambda x, y, z: z},
                   7: {"x": lambda x, y, z: y, "y": lambda x, y, z: x, "z": lambda x, y, z: 0},
                   8: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z, "z": lambda x, y, z: y},
                   # sextupoles
                   9: {"x": lambda x, y, z: x**2-y**2, "y": lambda x, y, z: -2*x*y, "z": lambda x, y, z: 0},
                  10: {"x": lambda x, y, z: 2*x*z, "y": lambda x, y, z: -2*y*z, "z": lambda x, y, z: x**2-y**2},
                  11: {"x": lambda x, y, z: z**2-y**2, "y": lambda x, y, z: -2*x*y, "z": lambda x, y, z: 2*x*y},
                  12: {"x": lambda x, y, z: 0, "y": lambda x, y, z: -2*y*z, "z": lambda x, y, z: z**2-y**2},
                  13: {"x": lambda x, y, z: 2*x*y, "y": lambda x, y, z: x**2-y**2, "z": lambda x, y, z: 0},
                  14: {"x": lambda x, y, z: y*z, "y": lambda x, y, z: x*z, "z": lambda x, y, z: x*y},
                  15: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z**2-y**2, "z": lambda x, y, z: 2*y*z},
                  # octupole
                  16: {"x": lambda x, y, z: x**3 - 3*x*y**2, "y": lambda x, y, z: y**3-3*x**2*y, "z": lambda x, y, z: 0},
                  17: {"x": lambda x, y, z: 3*x**2*z-3*z*y**2, "y": lambda x, y, z: -6*x*y*z, "z": lambda x, y, z: x**3 - 3*x*y**2},
                  18: {"x": lambda x, y, z: 3*x*z**2-3*x*y**2, "y": lambda x, y, z: -3*x**2*y-3*z**2*y+2*y**3, "z": lambda x, y, z: 3*x**2*z-3*z*y**2},
                  19: {"x": lambda x, y, z: z**3-3*z*y**2, "y": lambda x, y, z: -6*x*y*z, "z": lambda x, y, z: 3*x*z**2 - 3*x*y**2},
                  20: {"x": lambda x, y, z: 0, "y": lambda x, y, z: y**3-3*z**2*y, "z": lambda x, y, z: z**3-3*z*y**2},
                  21: {"x": lambda x, y, z: 3*x**2*y-y**3, "y": lambda x, y, z: x**3-3*x*y**2, "z": lambda x, y, z: 0},
                  22: {"x": lambda x, y, z: 6*x*y*z, "y": lambda x, y, z: 3*x**2*z-3*z*y**2, "z": lambda x, y, z: 3*x**2*y-y**3},
                  23: {"x": lambda x, y, z: 3*z**2*y-y**3, "y": lambda x, y, z: 3*x*z**2-3*x*y**2, "z": lambda x, y, z: 6*x*y*z},
                  24: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z**3-3*z*y**2, "z": lambda x, y, z: 3*z**2*y-y**3},
                 }

    def B_field(self, x=0, y=0, z=0):
        """Evaluates magnetic field at position x, y, z

        Parameters:
        * x: float, x position
        * y: float, y position
        * z: float, z position

        Returns:
        * array of length 3,  Magnetic field at position (x,y,z)
        """
        Bx = 0
        By = 0
        Bz = 0
        for i in self.P.keys():
            Bx += self.An[i]*self.P[i]["x"](x, y, z)
            By += self.An[i]*self.P[i]["y"](x, y, z)
            Bz += self.An[i]*self.P[i]["z"](x, y, z)
        return [Bx, By, Bz]

    def __call__(self, x=0, y=0, z=0):
        """Evaluates magnetic field at position x, y, z

        Parameters:
        * x: float, x position
        * y: float, y position
        * z: float, z position

        Returns:
        * array of length 3,  Magnetic field at position (x,y,z)
        """
        return self.B_field(x, y, z)

    def strength_to_str(self, multipole, strength):
        """Pretty string for multipole strength.

        Parameters:
        * multipole: int, Number of the multipole, allowed range 1 - 24.
        * strength: float, relative strength of the multipole at 1 cm distance

        Returns:
        * string giving type, of multipole, strength of gradient and shape of multipole
        """
        str = "%.1f ppm"%(strength/ppm)
        if strength < 1*ppm:
            str = "%.1f ppb"%(strength/ppb)

        vec = self.multipole_vector_str(multipole)

        if 1<=multipole and multipole <= 3:
            return "Dipole: %s$\cdot %s^T$"%(str, vec)

        if 4<=multipole and multipole <= 8:
            return "Quadrupole: %s/cm$\cdot %s^T$"%(str, vec)

        if 9<=multipole and multipole <= 15:
            return "Sextupole: %s/cm$^2\cdot %s^T$"%(str, vec)

        if 16<=multipole and multipole <= 24:
            return "Octupole: %s/cm$^3\cdot %s^T$"%(str, vec)

    def multipole_vector_str(self, multipole):
        """String representation of a multipole

        Parameters:
        * multipole: int, number of multipole, allowed range 1 - 24

        Returns:
        * string, shape of multipole
        """
        if multipole==1: return "(1, 0, 0)"
        if multipole==2: return "(0, 1, 0)"
        if multipole==3: return "(0, 0, 1)"

        if multipole==4: return "(x, -y, 0)"
        if multipole==5: return "(z, 0, x)"
        if multipole==6: return "(0, -y, z)"
        if multipole==7: return "(y, x, 0)"
        if multipole==8: return "(0, z, y)"

        if multipole==9: return "(x^2-y^2, -2xy, 0)"
        if multipole==10: return "(2xz, -2yz, x^2-y^2)"
        if multipole==11: return "(z^2-y^2, -2xy, 2xy)"
        if multipole==12: return "(0, -2yz, z^2-y^2)"
        if multipole==13: return "(2xy, x^2-y^2, 0)"
        if multipole==14: return "(yz, xz, xy)"
        if multipole==15: return "(0, z^2-y^2, 2yz)"

        if multipole==16: return "(x^3-3xy^2, y^3-3x^2y,0)"
        if multipole==17: return "(3x^2z-3zy^2, -6xyz, x^3 - 3xy^2)"
        if multipole==18: return "(3xz^2-3xy^2, -3x^2y-3z^2y+2y^3, 3x^2z-3zy^2)"
        if multipole==19: return "(z^3-3zy^2, -6xyz, 3xz^2 - 3xy^2)"
        if multipole==20: return "(0, y^3-3z^2y, z^3-3zy^2)"
        if multipole==21: return "(3x^2y-y^3, x^3-3xy^2, 0)"
        if multipole==22: return "(6xyz, 3x^2z-3zy^2, 3x^2y-y^3)"
        if multipole==23: return "(3z^2y-y^3, 3xz^2-3xy^2, 6xyz)"
        if multipole==24: return "(0, z^3-3zy^2, 3z^2y-y^3)"

    def multipole_name(self, multipole):
        """Returns type of multipole as string

        Parameters:
        * multipole: int, number of multipole, allowed range 1 - 24

        Returns:
        * string, type of multipole
        """
        if 1 <= multipole and multipole <= 3:
            return "Dipole"
        if 4 <= multipole and multipole <= 8:
            return "Quadrupole"
        if 9 <= multipole and multipole <= 15:
            return "Sextupole"
        if 16 <= multipole and multipole <= 24:
            return "Octupole"
        raise ValueError("Multipoles are only defined for index 1 to 24.")

    def set_strength_at_1cm(self, multipole, strength):
        """Calculates DeltaB from multipole at 1 cm distance. Takes different
        units from different multipoles into account

        Parameters:
        * multipole: int, number of multipole, allowed range 1 - 24
        * strength: float, strength of gradient
        """

        if  multipole < 1 or multipole > 24:
            raise ValueError("Multipoles are only defined for index 1 to 24.")
        elif 4 <= multipole and multipole <= 8:
            strength /= cm
        elif 9 <= multipole and multipole <= 15:
            strength /= cm**2
        elif 16 <= multipole and multipole <= 24:
            strength /= cm**3
        self.An[multipole] = strength*self.An[2]
