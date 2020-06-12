JOB=$1
sacct -j $JOB --format=JobID,Elapsed,nnodes,ncpus
