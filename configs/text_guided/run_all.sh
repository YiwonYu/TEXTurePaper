#!/bin/bash 
# command : bash configs/text_guided/run_all.sh
# List of YAML files
configs=(
    "alien_1.yaml"
    "alien_2.yaml"
    "alien_3.yaml"
    "alien_4.yaml"
    "alien_5.yaml"
    "rabbit_1.yaml"
    "rabbit_2.yaml"
    "rabbit_3.yaml"
    "rabbit_4.yaml"
    "rabbit_5.yaml"
    "sphere_1.yaml"
    "sphere_2.yaml"
    "sphere_3.yaml"
    "sphere_4.yaml"
    "sphere_5.yaml"
)

# Loop through each config and run the command
for config in "${configs[@]}"
do
    python -m scripts.run_texture --config_path=configs/text_guided/$config
done