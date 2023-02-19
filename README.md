# LEP-AD
Repurposing ESM pretrained models for DTI 

to get the data use this:
https://drive.google.com/drive/folders/1YaCspHVJCFdY-UCUyNrw0EtWqVfzmqrO?usp=share_link

It is recommended to use CUDA 11.4 version.
> conda env create -f environment1.yml
> 
> conda activate esm2

run data_protein_esm.ipynb notebook to create protein representations from ESM-2

> conda env create -f environment2.yml
> 
> conda activate LEP-AD

run LEP-AD.ipynb notebook to reproduce the results for each dataset
