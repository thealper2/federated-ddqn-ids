Server:

```shell
python frl_ids.py server --num-rounds 20 --data-dir /mnt/d/Datasets/nsl-kdd --min-available-clients 3
```

Clients (different terminals):

```shell
python frl_ids.py client --partition-id 0 --num-partitions 3 --data-dir /mnt/d/Datasets/nsl-kdd
python frl_ids.py client --partition-id 1 --num-partitions 3 --data-dir /mnt/d/Datasets/nsl-kdd
python frl_ids.py client --partition-id 2 --num-partitions 3 --data-dir /mnt/d/Datasets/nsl-kdd
```
