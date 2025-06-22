#!/bin/bash


# Check that input.xml exists
if [ ! -f input.xml ]; then
    echo "ERROR: input.xml not found in current directory." >&2
    exit 1
fi

# Lambda values and corresponding f-socket indices
lambdas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
socket_ids=(0 1 2 3 4 5 6 7 8 9 10)

# Loop over all lambda values
for i in "${!lambdas[@]}"; do
    lambda=${lambdas[$i]}
    sock_id=${socket_ids[$i]}
    dirname="${lambda}"
    seed=$RANDOM
    mkdir -p "$dirname"

    # Copy shared files into the directory
    cp hessian.data ref.data input.xml geop-RESTART "$dirname/"

    # Compute weights
    weight_anharmonic="$lambda"
    weight_harmonic=$(awk -v l="$lambda" 'BEGIN{printf "%.1f", 1.0 - l}')
    
    # Modify input.xml inside the folder
    sed -i -e "s/@ADDRESS/f$sock_id/" \
        -e "s/@RANDOM/$seed/" \
        -e "s/@HWEIGHT/$weight_harmonic/" \
        -e "s/@AWEIGHT/$weight_anharmonic/" \
            "$dirname/input.xml"

    # Also prepare the batch script
    new_address="f${sock_id}"
    sed "s/ADDRESS/$new_address/g" run_classical_ti.slurm > "${dirname}/run_classical_ti.slurm"

    echo "Prepared $dirname with lambda=$lambda and socket f${sock_id}"
done
