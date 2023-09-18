# LEP-AD
**Repurposing ESM Pretrained Models for Drug-Target Interaction (DTI)**

## Setting Up ESM-2 Repository

Begin by cloning the ESM-2 repository:

>
> git clone https://github.com/facebookresearch/esm.git
>

After cloning, navigate to the `esm` directory. Here, you'll need to create a directory for data storage:

>
> mkdir data
>

Next, download the required datasets from the provided link and ensure they are stored in the `data` directory you just created:
[Download Data](https://drive.google.com/drive/folders/1YaCspHVJCFdY-UCUyNrw0EtWqVfzmqrO?usp=share_link)

## Environment Configuration

For optimal performance, it's recommended to utilize CUDA 11.4. To set up the ESM environment, execute the following commands:

>
> conda env create -f environment.yml
>
> conda activate esm2
>

## Protein Representation with ESM

To derive protein representations from ESM, utilize the provided notebook. This will help in extracting unique proteins and making inferences using the ESM-2 model:

Execute the `data_protein_esm.ipynb` notebook to generate protein representations from ESM-2.

## LEP-AD for Drug-Target Interaction

With the protein representations from ESM in place, you're set to use LEP-AD for Drug-Target Interaction. To ensure there's no interference with the previous environment, we'll establish a new one:

>
> conda env create -f environment_LEP_AD.yml
>
> conda activate LEP-AD
>

To reproduce the results for each dataset, run the `LEP-AD.ipynb` notebook. Alternatively, the following command line can be executed:

>
> python LEP-AD.py --dataset Stitch --batch-size 128 --output_dim 64 --heads 4 --epochs 400 --n_layers 3 --step_size 250
>

