# meerpower
Code repo for power spectrum analysis pipeline

## Installation:

``git clone git@github.com:/meerklass/meerpower.git``

Creating a virtual environement is the easiest way to install dependencies and run this pipeline. If using ilifu, anaconda can be added using e.g.:

``module load anaconda3/2021.05``

Then a virtual environment created for meerpower with the required dependencies with <br /> - **Note:** this may take some time ~5-10 minutes

```
conda env create -f environment.yml
conda activate meerpower_env

# for optional jupyter support:
conda install -c anaconda ipykernel
ipython kernel install --name "meerpower_kernel" --user
```
