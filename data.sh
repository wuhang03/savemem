hf download JoeLeelyf/OVO-Bench --repo-type dataset --local-dir ./ovo
python unzip_ovo.py

hf download mjuicem/StreamingBench --repo-type dataset --local-dir ./StreamingBench
python unzip_streaming.py

hf download MCG-NJU/ODV-Bench --repo-type dataset --local-dir ./odv
python unzip_odv.py
