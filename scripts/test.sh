#!/usr/bin/env bash
echo "Dataset:" $1  "Backbone:" $2  "GPU index:" $3 "Tag:" $4 "logits_coeffs" $5

python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 3 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 7 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'



# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,0,0,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,0,0,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,0,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,0,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,0,0,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,0,0,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,0,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,0,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,3,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,3,2,0'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

# python exp2_test_few_shot.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='bcd,bcd,bcd,bcd' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
