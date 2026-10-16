#!/bin/bash -l

# SLURM options:

#SBATCH --job-name=polar_extract    # Nom du job
#SBATCH --output=/sps/grand/jlavoisier/code/quantification/polarization_cut/out_files/%j_SWF_chi2.log   # Standard output et error log

#SBATCH --partition=htc          # Choix de partition (htc par défaut)

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=8000                    # Mémoire en MB par défaut
#SBATCH --time=7-00:00:00             # Délai max = 7 jours

#SBATCH --mail-user=lavoisie@iap.fr   # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :


conda activate grandlib_test

python /sps/grand/jlavoisier/code/quantification/polarization_cut/2_extract_polar.py