# This Python file uses the following encoding: utf-8

# Implementation of the Pumpistor model of a flux pumped SQUID in the
# three wave mixing degenerate case ω_p = ω_s + ω_i and ω_p = ω_s.
#
# Based on an article from K. M. Sundqvist et al:
# "The pumpistor: A linearized model of a flux-pumped superconducting
# quantum interference device for use as a negative-resistance parametric
# amplifier"
# APL 109 102603 (2013),
# and on an article from J. Y. Mutus et al:
# "Design and characterization of a lumped element single-ended
# superconducting microwave parametric amplifier with on-chip
# flux bias line"
# APL 103 122602 (2013)

# Copyright (C) 2016 Dumur Étienne
# etienne.dumur@gmail.com

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import numpy as np
import scipy.constants as cst
from scipy.integrate import quad


class Transducer(object):



    def __init__(self, finger_width, finger_pitch, transducer_width,
                 nb_finger,
                 epsilon_11, epsilon_33, epsilon_13,
                 velocity_free, velocity_metallized,
                 transducer_type='single-electrode'
                 ):
        """
        Transducer model based on the Morgan book

        Attributes
        ----------
        finger_width : float
            Finger width of the transducer in meter.
        finger_pitch : float
            Distance between two fingers in meter.
        transducer_width : float
            Width of the transducer fingers in meter.
        nb_finger : int, float
            Number of finger in the transducer.
        epsilon_11, epsilon_33, epsilon_13: float
            Relative permitivity allong specific crystallographic axis.
        velocity_free : float
            Wave velocity in crystal without metallic layer on the surface in
            meter per second
        velocity_metallized : float
            Wave velocity in crystal with metallic layer on the surface in
            meter per second
        transducer_type : {'single-electrode', 'double-electrode'} str optional
            Type of the transducer, default single electrode.

        """

        self.finger_width     = finger_width
        self.finger_pitch     = finger_pitch
        self.transducer_width = transducer_width

        self.nb_finger = nb_finger

        self.epsilon_11 = epsilon_11
        self.epsilon_33 = epsilon_33
        self.epsilon_13 = epsilon_13

        self.velocity_free       = velocity_free
        self.velocity_metallized = velocity_metallized

        self.transducer_type = transducer_type



    @property
    def transducer_type(self):

        if self.Se == 2:
            return 'single-electrode'
        elif self.Se == 4:
            return 'double-electrode'

    @transducer_type.setter
    def transducer_type(self, transducer_type):
        """
        Return the parameter Se depending of the transducer type.

        Parameters
        ----------
        transducer_type : {single-electrode, double-electrode} str
            Transducer type

        Return
        ----------
        Se : {2., 4.} float
            Transducer type encoded in float
        """

        if type(transducer_type) is not str:
            raise TypeError('Parameter "transducer_type" must be string type.')

        if transducer_type == 'single-electrode':
            self.Se = 2.
        elif transducer_type == 'double-electrode':
            self.Se = 4.
        else:
            raise ValueError('The type of the transducer must be: "single-electrode" or "double-electrode".')




    def epsilon_inf(self):
        """
        The parameter epsilon_inf is defined such that the capacitance of a
        unit-aperture single-electrode transducer, per period, is simply
        epsilon_inf.
        This applies when the electrode widths equal the interelectrode gap
        widths.
        """

        return cst.epsilon_0*(1. + np.sqrt(self.epsilon_11*self.epsilon_33\
                                 - self.epsilon_13**2.))



    def alpha(self, M):
        """
        Return the alpha coeficient used for the calculation of the
        admittance of an uniform transducer.

        Parameters
        ----------
        M : int, float
            Harmonics number
        """

        if self.Se == 2 or self.Se == 3:
            return 1.
        elif self.Se == 4 :
            if M%2 == 1:
                return 2.
            else:
                return 0.



    def delta(self):
        """
        Dimensionless ratio of the electrode widths over the interelectrode gap.
        """

        return np.pi*self.finger_width/self.finger_pitch



    def gamma_s(self):
        """
        Ratio of the piezoelectric coupling over the epsilon_inf.
        """

        return (self.velocity_free - self.velocity_metallized)\
               /self.velocity_free/self.epsilon_inf()



    def legendre_function(self, nu, cosdelta):
        """
        Return the Legendre function P_nu(cosdelta).

        Parameters
        ----------
        nu : float, np.ndarray
            Order
        cosdelta : float
            parameter
            As defined in the Morgan this method expect the cosinus of the
            delta value, not the delta value itself.
        """

        delta = np.arccos(cosdelta)

        if delta < 0 or delta > np.pi:
            raise ValueError('pi*finger_width/finger_pitch must be comprised between 0 and pi.')

        def integrand(phi, nu, delta):
            """
                Integrand used in the scipy quad function.
            """

            return np.cos((nu + 1./2.)*phi)\
                  /np.sqrt(np.cos(phi) - cosdelta)

        # If nu is a list we make a loop along all its values and return the
        # result as an array
        if type(nu) is np.ndarray:

            temp = []
            for i in nu:
                a = quad(integrand,
                         -abs(delta), abs(delta), args=(i, delta))[0]
                temp.append(a)

            return np.array(temp)/np.pi/np.sqrt(2.)
        # if nu is a float, we just have to return the function
        else:
            return quad(integrand,
                       -abs(delta), abs(delta), args=(nu, delta))[0]\
                   /np.pi/np.sqrt(2.)



    def center_angular_frequency(self):
        """
        Return the central response frequency of the IDT in rad.Hz.
        """

        return 2.*np.pi*self.velocity_free/self.finger_pitch/self.Se



    def conductance_central(self, M):
        """
        Return the acoustic central conductance of the transducer.

        Parameters
        ----------
        M : int, float
            Harmonics number
        """

        return self.alpha(M)*M*self.center_angular_frequency()\
               *(self.nb_finger/2.)**2.*self.transducer_width*self.gamma_s()\
               *(2.*self.epsilon_inf()*np.sin(np.pi*M/self.Se)\
               /self.legendre_function(-M/self.Se, -np.cos(self.delta())))**2.



    def conductance(self, M, f):
        """
        Return the acoustic conductance of the transducer.

        Parameters
        ----------
        M : int, float
            Harmonics number
        f : float
            Frequency in Hz
        """

        X = np.pi*self.nb_finger/2.\
            *(2.*np.pi*f - self.center_angular_frequency())\
            /self.center_angular_frequency()

        return self.conductance_central(M)*(np.sin(X)/X)**2.



    def susceptance(self, M, f):
        """
        Return the acoustic conductance of the transducer.

        Parameters
        ----------
        M : int, float
            Harmonics number
        f : float
            Frequency in Hz
        """

        X = np.pi*self.nb_finger/2.\
            *(2.*np.pi*f - self.center_angular_frequency())\
            /self.center_angular_frequency()

        return self.conductance_central(M)*(np.sin(2.*X) - 2.*X)/2./X**2.



    def gamma(self):
        """
        Return the parameter gamma used in the capacitance calculation.
        """

        if self.Se == 2:
            return 1
        elif self.Se == 3:
            return 4./3
        else:
            return 2.



    def capacitance(self):
        """
        Return the electrical capacitance of the transducer.
        """

        return self.gamma()*self.transducer_width*self.nb_finger/2.\
               *self.epsilon_inf()*np.sin(np.pi/self.Se)\
               *self.legendre_function(-1./self.Se, np.cos(self.delta()))\
               /self.legendre_function(-1./self.Se, -np.cos(self.delta()))



    def electrical_Q_factor(self, M):
        """
        Return the electrical quality factor.

        Parameters
        ----------
        M : int, float
            Harmonics number
        """

        return M*self.center_angular_frequency()*self.capacitance()\
               /self.conductance_central(M)



    def elementary_charge_density(self, nb_finger, nb_point, out_finger=0.):
        """
        Return the charge density of a transducer when only one electrode
        (the central one) is polarized with 1 volt.

        Parameters
        ----------
        nb_finger : int
            Number of finger used for the calculation.
            The length of the transducer is calculated from the width and pitch
            of the fingers.
        nb_point : int
            Discretization of the simulation.
        out_finger {0, np.nan}, optional
            Choice to have the value 0 or np.nan in the output array when
            no electrode

        Return
        ----------
        x : np.ndarray
            x-axis along wich the charge density has been calculated.
            Return in unit of length/pitch in the purpose to have each finger
            center on interger.
        rho_f : np.ndarray
            Elementary charge density.

        Raises
        ------
        ValueError
            If the parameters are not in the good type
    """

        if type(nb_finger) is not int:
            raise ValueError('nb_finger must be an integer.')

        if type(nb_point) is not int and type(nb_point) is not float:
            raise ValueError('nb_point must be an integer.')

        if nb_finger < 0:
            raise ValueError('nb_finger must be positive.')

        if nb_point < 0:
            raise ValueError('nb_point must be positive.')

        if out_finger not in (0, np.nan) :
            raise ValueError('out_finger must be "0" or "np.nan".')

        # Due to symmetry rule, we can calculate the positive value only
        # and get the negative by mirror symmetry
        x = np.linspace(0., self.finger_pitch*nb_finger, nb_point)

        def gamma(theta, delta):

            def integrand(s, theta, delta):

                return np.sin(np.pi*s)*np.cos((s - 1./2.)*theta)\
                       /self.legendre_function(-s, -np.cos(delta))

            return quad(integrand, 0., 1., args=(theta, delta))[0]

        temp = []
        for i in x:

            m = np.rint(i/self.finger_pitch)

            if abs(i - m*self.finger_pitch) < self.finger_width/2. :

                theta = 2.*np.pi*i/self.finger_pitch

                temp.append(self.epsilon_inf()/self.finger_pitch\
                       *2.*np.sqrt(2)*(-1.)**m*gamma(theta, self.delta())\
                       /np.sqrt(np.cos(theta) - np.cos(self.delta())))
            else:
                temp.append(out_finger)

        # get total x and rho_f by symmetry
        x     = np.concatenate((-x[1  :][::-1], x))/self.finger_pitch
        rho_f = np.concatenate((temp[1:][::-1], temp))

        return x, rho_f



    def charge_density(self, nb_finger, nb_point, out_finger=0):
        """
        Return the charge density of a transducer when one on two electrodes
        are polarized with 1 volt.

        Parameters
        ----------
        nb_finger : int
            Number of finger used for the calculation.
            The length of the transducer is calculated from the width and pitch
            of the fingers.
        nb_point : int
            Discretization of the simulation.
        out_finger {0, np.nan}, optional
            Choice to have the value 0 or np.nan in the output array when
            no electrode

        Return
        ----------
        x : np.ndarray
            x-axis along wich the charge density has been calculated.
            Return in unit of length/pitch in the purpose to have each finger
            center on interger.
        rho_e : np.ndarray
            Charge density.

        Raises
        ------
        ValueError
            If the parameters are not in the good type
        """

        if int((nb_point*2. - 2.)/nb_finger) != (nb_point*2. - 2.)/nb_finger:
            correct = int((round((nb_point*2.-2.)/nb_finger)*nb_finger+2.)/2.)
            raise ValueError("Your nb_point doesn't fulfill required for the"
                             "use of the Superposition Theorem. Try using"
                             "nb_point="+str(correct)+".")

        # out_finger = 0 whatever the user ask for since we need 0 for following
        # calculations
        x, rho_f = self.elementary_charge_density(nb_finger, nb_point,
                                                  out_finger=0.)

        # For symmetry reason we don't erase the last point of the array
        # If we don't this point will be more superposed than the other.
        x     = x[:-1]
        rho_f = rho_f[:-1]

        # We superposed all the elementary charge density created by the
        # electrode
        rho_e = np.zeros_like(rho_f)
        for i in range(nb_finger):
            print i
            rho_e += np.roll(rho_f, int(len(rho_f)/nb_finger*i))

        # When no electrode the calculation return 0.
        # We replace this value by np.nan if the user ask so.
        rho_e[rho_e==0] = out_finger

        return x, rho_e
