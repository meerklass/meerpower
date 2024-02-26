# meerpower
Code repo for power spectrum analysis pipeline

See ``demo/demo_notebook.ipynb`` [not up-to-date for new kernel] for an example notebook with some analysis capabilities.

## Installation:

``git clone git@github.com:/meerklass/meerpower.git``

If using **ilifu**, you should be able to run Jupyter notebooks by creating an environment within JupyterLab.

Compatible anaconda environment can be created on ilifu by first adding anaconda:

``module load anaconda3/2021.05``

Then create a virtual environment with the required dependencies by following the below. This currently uses *gridimp* environment (https://github.com/stevecunnington/gridimp) <br /> - **Note:** this may take some time ~5-10 minutes

```
conda env create -f environment_gridimp.yml # [previously environment.yml]
conda activate gridimp # [previously meerpower_env]

# for optional jupyter support:
python -m pip install ipykernel
python -m ipykernel install --name "gridimp" --user
```
