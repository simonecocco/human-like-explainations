# PattaLM

## Step 1
Go into pathlm directory and run `pip3 install .`

## Step 2
`chmod u+x create_dataset.sh`

`create_dataset.sh ml1m 1000 3 8`

## Step 3
`python3 pathlm/models/lm/fill_template.py -L 1000 `

## Step 4
`python3 pathlm/datasets/tokenize_dataset.py --only-patta`

## Step 5
`python3 pathlm/models/lm/patta_main.py --epochs 10 --eval 'U169 R-1'`