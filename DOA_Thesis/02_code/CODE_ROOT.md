# Code Root

Active implementation directory:

`D:\AI\DOA_est_Master\myDOA`

Do not copy the full codebase into this thesis directory. Keep thesis materials here and run code from `myDOA`.

## Main Code Mapping

| Thesis topic | Code path |
| --- | --- |
| Four-channel SCM input | `data/signal_datasets.py` |
| Coordinate attention | `models/coord_attention.py` |
| CA-DOA-Net | `models/ca_doa_net.py` |
| Vanilla CNN baseline | `models/vanilla_cnn.py` |
| Spectrum peak search | `models/base_network.py`, `test/compare_methods.py` |
| Loss functions | `utils/loss_function.py` |
| Training CA-DOA-Net | `train/train_ca_doa.py` |
| Training Vanilla CNN | `train/train_vanilla_cnn.py` |
| SNR test | `test/test_snr.py` |
| Method comparison | `test/compare_methods.py` |

