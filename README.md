# CNPEM-MPI Joint Meeting 2025: Tutorial on Nuclear Quantum Effects

## Overview

In this tutorial you will learn how to calculate thermodynamic properties at thermal equilibrium using i-PI. You will include nuclear quantum effects into your calculations by running path-integral molecular dynamics (PIMD). In the first part, you will compute the equilibrium density of low-temperature para-hydrogen by running an NpT simulation. In the second part, you will explore different strategies for calculating quantum free energy differences.

## Resources

  - Webpage of our research group: https://www.mpsd.mpg.de/research/groups/sabia
  - GitHub repository of this tutorial: https://github.com/sabia-group/i-PI-tutorial
  - i-PI documentation: https://docs.ipi-code.org/

## Installation

Install the python packages, in a virtual environment if need be:

```bash
python -m pip install -r requirements.txt
```

Put the tutorial modules into the system path

```bash
source env.sh
```

If you haven't already, install i-pi to a convenient location (choose a suitable path instead of `/path/to/i-pi`):

```bash
git clone git@github.com:i-pi/i-pi.git /path/to/i-pi
```

Compile the drivers...

```bash
cd /path/to/i-pi
cd drivers/f90
make
cd ../..
```

...and make sure i-PI is also in the system path:

```bash
source /path/to/i-pi/env.sh
```

Now you're ready to go!
