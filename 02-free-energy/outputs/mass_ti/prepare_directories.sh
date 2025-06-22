#!/bin/bash


# Check that input.xml exists
if [ ! -f input.xml ]; then
    echo "ERROR: input.xml not found in current directory." >&2
    exit 1
fi

# Lambda values and corresponding f-socket indices
lambdas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
socket_ids=(1 2 3 4 5 6 7 8 9 10)
nbeads=(4 4 8 8 12 12 12 16 16 16)
N=16
tau0="20"
dt0="1.0"
m0="1837.4"

# Loop over all lambda values
for i in "${!lambdas[@]}"; do
    lambda=${lambdas[$i]}
    N=${nbeads[$i]}
    sock_id=${socket_ids[$i]}
    dirname="${lambda}"
    seed=$RANDOM
    mkdir -p "$dirname"
    # Copy shared files into the directory
    cp init.xyz input.xml $dirname/
    # Update time constants
    dt=$(awk -v x="$dt0" -v l="$lambda" 'BEGIN{printf "%.1f", x/l}')
    tau=$(awk -v x="$tau0" -v l="$lambda" 'BEGIN{printf "%.1f", x/l}')
    # Update mass
    mass=$(awk -v l="$lambda" -v m="$m0" 'BEGIN{printf "%.2f", m / (l*l)}')

    # Modify input.xml inside the folder
    sed -i -e "s/@ADDRESS/f$sock_id/" \
        -e "s/@RANDOM/$seed/" \
        -e "s/@NBEADS/$N/" \
        -e "s/@MASS/$mass/" \
        -e "s/@DT/$dt/" \
        -e "s/@TAU/$tau/" \
            "$dirname/input.xml"

    # Also prepare the batch script
    new_address="f${sock_id}"
    sed "s/ADDRESS/$new_address/g" run_mass_ti.slurm > "${dirname}/run_mass_ti.slurm"
    sed -i "s/@NBEADS/$N/g" "${dirname}/run_mass_ti.slurm"
    
    echo "Prepared $dirname with lambda=$lambda and socket f${sock_id}"

done
