#!/bin/bash

# Directory for the output .out files of the SLURM jobs
output_dir_out="/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts/out"
mkdir -p "$output_dir_out"

# Path to the temporary text file containing the list of files
mypath="/sps/grand/jlavoisier/code/quantification/check_data/filtered_files.txt"


# Read the list of files from the text file
mapfile -t listeoffiles < "$mypath"

# Number of files to process
n=20000

# Get the list of files
# mapfile -t listeoffiles < <(ls "$dir_path" | grep -E 'CD') 
# mapfile -t listeoffiles < <(ls "$dir_path" | grep -E 'CD.*51DUs') # if we want to process only the files with x DUs
listeoffiles=("${listeoffiles[@]:0:n}") # Take the first n files

# Files per job
files_per_job=100

# Calculate the number of jobs
num_jobs=$(( (${#listeoffiles[@]} + files_per_job - 1) / files_per_job ))

# Loop over batches of files
for ((job=0; job<num_jobs; job++)); do
    start_idx=$((job * files_per_job))
    end_idx=$((start_idx + files_per_job - 1))

    # Ensure we don't exceed the total number of files
    if [ $end_idx -ge ${#listeoffiles[@]} ]; then
        end_idx=$(( ${#listeoffiles[@]} - 1 ))
    fi

    # Extract the files for this job
    job_files=(${listeoffiles[@]:$start_idx:$files_per_job})

    # Generate a unique job name
    job_name="adf_batch_${job}.bash"

    # Create the SLURM script
    echo "#!/bin/bash" > "$job_name"
    echo "#SBATCH -t 7-00:00 -n 1 --mem 9G" >> "$job_name"
    echo "#SBATCH -o $output_dir_out/batch_${job}.out" >> "$job_name"

    # Loop over the files in this batch
    for file in "${job_files[@]}"; do
        mkdir -p "$output_dir_out"
        echo "python /sps/grand/jlavoisier/code/quantification/polarization_cut/1_parasite_polarization.py $file" >> "$job_name"
    done

    # Submit the job
    sbatch "$job_name"
    rm "$job_name"
done