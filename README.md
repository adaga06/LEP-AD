# LEP-AD
Repurposing ESM pretrained models for DTI 
Start with cloning the ESM-2 Repor from 

>
> git clone https://github.com/facebookresearch/esm.git
>

Once you are in esm directory . To get the data use this to make data directory:
>
> mkdir data
>
download the data and keep various datasets in the above sepcificed data folder.
https://drive.google.com/drive/folders/1YaCspHVJCFdY-UCUyNrw0EtWqVfzmqrO?usp=share_link

It is recommended to use CUDA 11.4 version.
To create ESM environment use below command line
> conda env create -f environment.yml
> 
> conda activate esm2
to get Protein representation from ESM use notebook to get unique proteins and get the inference from ESM-2 model
run data_protein_esm.ipynb notebook to create protein representations from ESM-2

Once you have protein representation from ESM, you can use LEP-AD for Drug-Target Interaction.
For this we will create a new environment, to prevent disturbing old environment.

> conda env create -f environment_LEP_AD.yml
> 
> conda activate LEP-AD

run LEP-AD.ipynb notebook to reproduce the results for each dataset
