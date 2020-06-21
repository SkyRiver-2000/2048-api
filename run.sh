# Generate 15,000 log files in total
declare -i Num_Attempts=15000

# Create storage directory
if [ ! -d log_storage ]; then
    mkdir log_storage
fi

# From 1 to Num_Attempts,
# generate temp*.log in storage directory and display information
for i in $( seq 1 ${Num_Attempts} )
do
    python evaluate.py >> log_storage/temp${i}.log
    printf "Round:%4d/%4d, " ${i} ${Num_Attempts}
    tail -1 log_storage/temp${i}.log
done

# Compile and execute C++ code to get statistics
g++ -o main data_analysis.cpp
./main
