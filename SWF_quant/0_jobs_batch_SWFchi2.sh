#!/bin/bash
# Path to the directory containing the files
# ZHAiReS 
dir_path_AN="/sps/grand/DC2.1rc4/GP300ZHAireS-AN"
dir_path_NJ="/sps/grand/DC2.1rc4/GP300ZHAireS-NJ"

# #Coreas
# dir_path_AN="/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN"
# dir_path_NJ="/sps/grand/DC2_Coreas/Coreas_nonoise"


# Directory for the output .out files of the SLURM jobs
output_dir_out="/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/out"
mkdir -p "$output_dir_out"

# If the directory does not exist, exit the script
if [ ! -d "$dir_path_AN" ]; then
    echo "Le répertoire $dir_path_AN n'existe pas."
    exit 1
fi

if [ ! -d "$dir_path_NJ" ]; then
    echo "Le répertoire $dir_path_NJ n'existe pas."
    exit 1
fi

# Number of files to process
n=100

# Get the list of files
mapfile -t listeoffiles_AN < <(ls "$dir_path_AN" | grep -E 'CD')
mapfile -t listeoffiles_NJ < <(ls "$dir_path_NJ" | grep -E 'CD')
listeoffiles_AN=("${listeoffiles_AN[@]:0:n}") # Take the first n files
listeoffiles_NJ=("${listeoffiles_NJ[@]:0:n}") # Take the first n files


# Files per job
files_per_job=2

# Calculate the number of jobs
num_jobs=$(( (${#listeoffiles_AN[@]} + files_per_job - 1) / files_per_job ))

# Loop over batches of files
for ((job=0; job<num_jobs; job++)); do
    start_idx=$((job * files_per_job))
    end_idx=$((start_idx + files_per_job - 1))

    # Ensure we don't exceed the total number of files
    if [ $end_idx -ge ${#listeoffiles_AN[@]} ]; then
        end_idx=$(( ${#listeoffiles_AN[@]} - 1 ))
    fi

    # Extract the files for this job
    job_files_AN=(${listeoffiles_AN[@]:$start_idx:$files_per_job})
    job_files_NJ=(${listeoffiles_NJ[@]:$start_idx:$files_per_job})

    # Generate a unique job name
    job_name="SWF_part_${job}.bash"

    # Create the SLURM script
    echo "#!/bin/bash" > "$job_name"
    echo "#SBATCH -t 4-00:00 -n 1 --mem 20G" >> "$job_name"
    echo "#SBATCH -o $output_dir_out/batch_${job}.out" >> "$job_name"

    # Loop over the files in this batch
    for ((i=0; i<${#job_files_AN[@]}; i++)); do
        file_AN="${job_files_AN[$i]}"
        file_NJ="${job_files_NJ[$i]}"
        full_path_AN="$dir_path_AN/$file_AN"
        full_path_NJ="$dir_path_NJ/$file_NJ"
        echo "python /pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/build_SWFchi2_sims.py $full_path_AN $full_path_NJ" >> "$job_name"
    done

    # Submit the job
    sbatch "$job_name"
    rm "$job_name"
done