# Define resolution modes and padding values
RES_MODES=("matched" "all")
PADDING_VALUES=(0 128)

for RES_MODE in "${RES_MODES[@]}"; do
    for PAD in "${PADDING_VALUES[@]}"; do
        JOB_NAME="rezip_${RES_MODE}_${PAD}"
        rm -f "${JOB_NAME}.out"
        rm -f "${JOB_NAME}.err"
        bsub -n 24 -e "${JOB_NAME}.err" -o "${JOB_NAME}.out" -J $JOB_NAME sh ./zip.sh "$RES_MODE" "$PAD"
    done
done

# sleep 30  # Wait 30 seconds before checking the LSF queue

# while true; do
#     # Check for "rezip" jobs in the LSF queue
#     if bjobs | grep -q "rezip"; then
#         echo "Rezip jobs are still running..."
#     else
#         echo "No rezip jobs found."
#         break  # Exit the loop when no "rezip" jobs are left
#     fi
#     sleep 360  # Wait 360 seconds before checking again
# done
# echo "All done!"