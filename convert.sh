${HOME}/anaconda3/envs/lerobot/bin/python convert_lerobot_interleave.py \
  --repo-id "${HF_USER}/<dataset>" \
  --root ${HOME}/lerobot_datasets/<dataset> \
  --output ${HOME}/lerobot_datasets/<dataset>_interleaved \
  --objects "toy bucket of popcorn" "toy bottle of ketchup" \
  --device cuda \
  --camera-name SideLeft
