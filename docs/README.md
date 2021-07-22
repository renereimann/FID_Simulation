# General Description

The repository contains scripts to generate simulations of pulsed Nuclear Magnetic Resonance (pNMR) free induction decay (FID) signals.

## Installation & Requirements

This package is a pure python package and does not yet need a full installation. Just make sure you have installed the dependencies listed below and all scripts from this repository in a single folder.

The package could be installed using `pip`.
explain editable mode and user Installation

The package is based on
* numpy
* scipy
* numericalunits (<= numericalunits-1.23 if used with python2)
* matplotlib (for plotting)
* time, copy, json (from standard modules)
* ROOT (optional in one analysis mode)

## Description of Physics

The setup contains the following part:
* A magnetic field which should be measured using the NMR Probe. This field is called the [Background Magnetic Field](Physics.md#background_field).
* The NMR probe itself is made out of a [probe sample](Physics.md#probe) made out of some proton rich [material](Physics.md#materials) surrounded by a [probe coil](Physics.md#coil_field) which is used to apply [RF pulses](Physics.md#rf_pulse) and [pickup the induced signal](Physics.md#FreeInductionDecay) in the probe sample.

We use the effect of spin precession, which introduces a makroscopic measurable changing in magnetization. The changing magnetization induces an electromotive force and thus a signal in a pickup coil. Spin precession happens when the spins are not aligned with the external magnetic field. To initially get the spins out of their equalibrium state, which is aligned with the external magnetic field, an RF pulse is applied to the probe coil. In general RF pulses can be used to change the orientation of the spins.

1. [Background Magnetic Field](Physics.md#background_field)
2. [Coil Magnetic Field](Physics.md#coil_field)
3. [Probe Sample](Physics.md#probe)
4. [Materials and Properties](Physics.md#materials)
5. [Pulsed NMR sequences](Physics.md#sequence)
6. [Bloch Equations](Physics.md#bloch_eq)
7. [RF Pulse](Physics.md#rf_pulse)
8. [Free Induction Decay](Physics.md#FreeInductionDecay)
9. [Spin Echo](Physics.md#spin_echo)
10. [Noise](Physics.md#noise)
11. [Signal Artefacts](Physics.md#artefacts)

## Simulation Principle

## Analysis Strategies

## Examples

## Limitations of Approximations

## API Documentation

## ToDo List

see [the ToDo List](ToDoList.md)

##  References
[scipy] www.scipy.org
[numericalunits] www.github.com/numericalunits
