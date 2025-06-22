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
    mkdir -p "$dirname"

    # Copy shared files into the directory
    cp hessian.data geop-RESTART ref.data input.xml "$dirname/"

    # Compute weights
    weight_anharmonic="$lambda"
    weight_harmonic=$(awk -v l="$lambda" 'BEGIN{printf "%.6f", 1.0 - l}')

    # Modify input.xml inside the folder
    awk -v wh="$weight_harmonic" -v wa="$weight_anharmonic" -v fx="f$sock_id" '
    BEGIN { inside_forces=0; inside_ffsocket=0 }
    /<forces>/     { inside_forces=1; print; next }
    /<\/forces>/   { inside_forces=0;
                     print "    <force forcefield='\''harmonic'\'' weight='\''" wh "'\''> </force>";
                     print "    <force forcefield='\''anharmonic'\'' weight='\''" wa "'\''> </force>";
                     print; next }
    inside_forces  { next }

    /<ffsocket/    { inside_ffsocket=1; print; next }
    /<\/ffsocket>/ { inside_ffsocket=0; print; next }
    inside_ffsocket && /<address>/ {
        print "    <address> " fx " </address>"
        next
    }

    { print }
    ' "$dirname/input.xml" > "$dirname/input.tmp" && mv "$dirname/input.tmp" "$dirname/input.xml"

    echo "Prepared $dirname with lambda=$lambda and socket f${sock_id}"
done

