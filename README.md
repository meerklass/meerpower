# meerpower
Code repo for power spectrum analysis pipeline

See ``demo/demo_notebook.ipynb`` for an example notebook of some analysis capabilities.

## Installation:

``git clone git@github.com:/meerklass/meerpower.git``

If using **ilifu**, you should be able to run the demo Jupyter notebook by chosing the environment "meerpower_kernel" within Jupyter.

Alternatively, one can create their own virtual environement which is the easiest way to install dependencies and run this pipeline. For example on ilifu, anaconda can be added using e.g.:

``module load anaconda3/2021.05``

Then a virtual environment created for meerpower with the required dependencies with <br /> - **Note:** this may take some time ~5-10 minutes

```
conda env create -f environment.yml
conda activate meerpower_env

# for optional jupyter support:
conda install -c anaconda ipykernel
ipython kernel install --name "meerpower_kernel" --user
```
