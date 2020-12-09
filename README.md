about hyphae
=============

The Script generates root-like networks where no edges overlap. 
More details see website of the author: https://inconvergent.net/generative/hyphae/ 

## This Fork ##

This is a little clean-up of the original script. It is
- **python3**-compatible
- reduced to core functionality
- available with **conda-environment** 

**Output**: This creates a series of pictures in a directory called "out" in the working directory
They later have to be cut together to create "the" simulation

### Getting started ###

1. Clone this repository and change into it
```
git clone https://github.com/MerlinSzymanski/hyphae
cd hyphae
```

2. Create and activate the conda environment
```
conda env create -f environment.yml
conda activate hyphae
```

3. Run the script:
```
python3 hyphae.py
```
WARNING: The script never stops! Use ctrl+c to stop the script when you think it is enough

----
Anders Hoff 2014

