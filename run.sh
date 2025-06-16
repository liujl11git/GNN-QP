# Generate samples
export qalpha=0.95
export kvalid=100
export ktest=100

export lr=5e-4
export bw=50
export batchsize=500

ktrain=500
embsize=256
task=solution
gpu=0
seed=42
expname=""
mi_arg=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --ktrain)
      ktrain="$2"
      shift 2
      ;;
    --embsize)
      embsize="$2"
      shift 2
      ;;
    --task)
      task="$2"
      shift 2
      ;;
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --mixed-integer)
      expname="mi"
      mi_arg="--mixed-integer"
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

train_seed=$(($seed + 7))

export CUDA_VISIBLE_DEVICES=$gpu

expname+="lcqp"
data_path="./data/${expname}-${ktrain}-qsparse-seed${seed}"
save_path="./results/${expname}-${ktrain}-seed${seed}-gnn-emb${embsize}-${task}-lr${lr}-bw${bw}-seed${train_seed}"

if [ ! -d $data_path ]
then
    echo "Generating new instances in ${data_path}..."
    python generate_data.py \
        --q-alpha $qalpha \
        --k-train $ktrain --k-valid $kvalid --k-test $ktest \
        --feasible-only \
        --seed $seed \
        --path $data_path $mi_arg
    echo "Instance generation complete."
else
    echo "${data_path} exists. Using existing problem instances..."
fi


python train.py --model GNN \
	--data-path "${data_path}/train" \
	--valid-data-path "${data_path}/valid" \
	--save-path $save_path \
	--emb-size $embsize --task $task \
	--seed $train_seed \
	--lr $lr --best-wait $bw --batch-size $batchsize $mi_arg
