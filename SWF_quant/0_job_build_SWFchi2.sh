#!/bin/bash -l

# SLURM options:

#SBATCH --job-name=SWF_quant    # Nom du job
#SBATCH --output=/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/%j_SWF_chi2.log   # Standard output et error log

#SBATCH --partition=htc          # Choix de partition (htc par défaut)

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=20000                    # Mémoire en MB par défaut
#SBATCH --time=7-00:00:00             # Délai max = 7 jours

#SBATCH --mail-user=lavoisie@iap.fr   # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :


conda activate grandlib_test
# python /pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/build_SWFchi2_data.py

# python /pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/build_SWFchi2_sims.py
python /pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/build_SWFchi2_sims.py /sps/grand/DC2.1rc4/GP300ZHAireS-AN/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-AN_0012 /sps/grand/DC2.1rc4/GP300ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0012
