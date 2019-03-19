#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --tasks-per-node=15
#SBATCH -J asr_sim_active_lstm
#SBATCH --output=/Users/qubix/Documents/work/sim_asr/automated-systematic-review-simulations/sub_analysis_AL_LSTM/batch.slurm_lisa/asr_sim_active_lstm/asr_sim_active_lstm_0.out
#SBATCH --error=/Users/qubix/Documents/work/sim_asr/automated-systematic-review-simulations/sub_analysis_AL_LSTM/batch.slurm_lisa/asr_sim_active_lstm/asr_sim_active_lstm_0.err

module load eb
module load Python/3.6.1-intel-2016b

BASE_DIR=/Users/qubix/Documents/work/sim_asr/automated-systematic-review-simulations/sub_analysis_AL_LSTM
cd $BASE_DIR
mkdir -p "$TMPDIR"/output
rm -rf "$TMPDIR"/results.log
cp -r $BASE_DIR/pickle "$TMPDIR"
cd "$TMPDIR"
    
parallel -j 15 << EOF_PARALLEL
sleep 0; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results0.log
sleep 1; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results1.log
sleep 2; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results2.log
sleep 3; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results3.log
sleep 4; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results4.log
sleep 5; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results5.log
sleep 6; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results6.log
sleep 7; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results7.log
sleep 8; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results8.log
sleep 9; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results9.log
sleep 10; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results10.log
sleep 11; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results11.log
sleep 12; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results12.log
sleep 13; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results13.log
sleep 14; python3 -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy random  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results14.log
EOF_PARALLEL

cp -r "$TMPDIR"/output  $BASE_DIR

if [ "False" == "True" ]; then
    echo "Job $SLURM_JOBID ended at `date`" | mail $USER -s "Job: asr_sim_active_lstm/0 ($SLURM_JOBID)"
fi
date
