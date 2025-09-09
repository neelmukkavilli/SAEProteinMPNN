#!/bin/bash

# --show_img to show 3D PCA plot, do not use wihtout adjusting img_mask_level

input_csv='newsamples500_2'

python clustering.py \
        --csv $input_csv \
        --num_clusters 100 \
        --bottom_k 25 \
        --return_num_clusters 100 \
        --cluster_mask_level 0.1 \
        --img_mask_level 0.1 \
        --show_img

cd data_labeling/uniprot

Uniprot_query="true"

if [ "$Uniprot_query" == "true" ]; then
    mkdir -p out;
    for pdb in $(cat pdb_list.txt); do
        if [ ! -f "out/${pdb}.tsv" ]; then
            echo "$pdb"
            wget -q "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/$pdb.xml.gz" -O  - | gunzip | python parse_sifts.py 1> out/$pdb.tsv 2> /dev/null;
        fi
    done
fi

cd ../../comparison

# --automate to return max percentage for each cluster or stdv/mean

python collect_cluster_features.py \
    --csv $input_csv \
    --automate
