
import os
import pandas as pd
import json 
import subprocess
import torch
import pickle

dataset = 'davis'

test_data = pd.read_csv('data/'+dataset+'/'+dataset+'_test.csv')
train_data = pd.read_csv('data/'+dataset+'/'+dataset+'_train.csv')

complete_data=pd.concat([train_data,test_data],axis=0)

protein_set = list(set(complete_data['target_sequence']))
drug_set=list(set(complete_data['compound_iso_smiles']))
print(len(protein_set))
print(len(drug_set))

protein_dict= {}
for i in range(len(protein_set)):
    protein_dict[i] = protein_set[i]
with open('data/'+dataset+'/protein_dict.txt','w') as file:
    file.write(json.dumps(protein_dict))

drug_dict= {}
for i in range(len(drug_set)):
    drug_dict[i] = drug_set[i]
with open('data/'+dataset+'/drug_dict.txt','w') as file:
    file.write(json.dumps(drug_dict))

result=''
for key, value in protein_dict.items():
    result+='>UniRef50_'+str(key)+'\n'+str(value)+'\n'
f = open('data/'+dataset+'/proteins.fasta', 'w')
f.write(result)

if not os.path.isdir('data/'+dataset+'/'+'/proteins_emb_esm2'):
    os.mkdir('data/'+dataset+'/'+'/proteins_emb_esm2')
process = "python scripts/extract.py "+"esm2_t36_3B_UR50D"+" data/davis/proteins.fasta data/"+dataset+"/proteins_emb_esm2 --include mean per_tok"

subprocess.call(process, shell=True)

if not os.path.exists('data/'+dataset+'/temp/protein_rep.pickle'):
    esm_emb_path = 'data/' + dataset + '/proteins_emb_esm2/UniRef50_'
    protein_dict = json.load(open('data/davis/protein_dict.txt'))
    target_reps_dict = {}
    i=0
    for key in protein_dict.keys():
        target_reps_dict[protein_dict[key]] = torch.load(esm_emb_path+ key + '.pt')['mean_representations'][36]
        if i%1000==0:
            print(i)
        i+=1

    with open('data/'+ dataset +'/temp/protein_rep.pickle','wb') as handle:
        pickle.dump(target_reps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved protein representations')