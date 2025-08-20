mkdir -p out;
for pdb in $(cat pdb_list.txt); do
  if [ ! -f "out/${pdb}.tsv" ]; then
    echo $pdb
    wget -q "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/$pdb.xml.gz" -O  - | gunzip | python parse_sifts.py 1> out/$pdb.tsv 2> /dev/null;
  fi
done
