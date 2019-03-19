#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --tasks-per-node=10
#SBATCH -J asr_sim_active_lstm

module load eb
module load Python/3.6.1-intel-2016b

BASE_DIR=/Users/qubix/Documents/work/sim_asr/automated-systematic-review-simulations/sub_analysis_AL_LSTM
cd $BASE_DIR
mkdir -p "$TMPDIR"/output
rm -rf "$TMPDIR"/results.log
cp -r $BASE_DIR/pickle "$TMPDIR"
cd "$TMPDIR"
    
parallel -j 15 << EOF_PARALLEL
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results90.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results91.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results92.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results93.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results94.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results95.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results96.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results97.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results98.log &> /dev/null
python3 -m asr simulate pickle/schoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results99.log &> /dev/null
EOF_PARALLEL

cp -r "$TMPDIR"/output  $BASE_DIR

if [ "False" == "True" ]; then
    echo "Job $SLURM_JOBID ended at `date`" | mail $USER -s "Job: asr_sim_active_lstm/6 ($SLURM_JOBID)"
fi
date
