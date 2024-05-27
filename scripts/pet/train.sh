
CONFIG=$1
cfg_path="configs/pet/$CONFIG"

python ./tools/train_cc.py  \
--cfg=$cfg_path \
--local-rank=0 \
--launcher="pytorch" \
--debug=True