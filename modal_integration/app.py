"""
Main Modal App for HRM Training

Single unified entrypoint. All functions registered on shared app from modal_app.
Usage:
    modal run modal_integration/app.py::run_download --dataset cifar10
    modal run modal_integration/app.py::run_train --dataset cifar10 --epochs 1 --push
"""
import modal as modal_client
import sys
from modal import App

# Ensure project root is on sys.path for Modal container
for p in ['/root/HRM', '/root', '/root/modal_integration']:
    if p not in sys.path:
        sys.path.insert(0, p)
from modal_integration.modal_app import app
from modal_integration import volumes, images, modal_secrets, train_modal
DEFAULT_SECRETS = [modal_client.Secret.from_name('hf-token'), modal_client.Secret.from_name('wandb-api-key')]
DEFAULT_VOLUMES = {'/data': volumes.datasets_volume, '/checkpoints': volumes.checkpoints_volume, '/root/.cache/huggingface': volumes.hf_cache_volume}

@app.function(image=images.hrm_base_image, volumes=DEFAULT_VOLUMES, secrets=DEFAULT_SECRETS, timeout=300)
def hello():
    import torch
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    return 'OK'

@app.function(image=images.hrm_base_image, volumes=DEFAULT_VOLUMES, secrets=DEFAULT_SECRETS, timeout=300)
def check(data='cifar10'):
    from pathlib import Path
    from modal_integration.volumes import DATASET_PATHS
    p = Path(DATASET_PATHS.get(data, f'/data/{data}'))
    return {'exists': p.exists(), 'files': len(list(p.iterdir())) if p.exists() else 0}

@app.function(image=images.hrm_base_image, volumes=DEFAULT_VOLUMES, secrets=DEFAULT_SECRETS, timeout=300)
def list_ckpts():
    from pathlib import Path
    d = Path('/checkpoints/models')
    ckpts = list(d.glob('*.pt')) if d.exists() else []
    return {'count': len(ckpts), 'names': [c.name for c in ckpts]}

@app.local_entrypoint()
def i_download(dataset='all', force=False):
    import modal_integration.train_modal as tm
    ds = None if dataset == 'all' else [dataset]
    result = tm.download_datasets.remote(ds, force)
    print(result)
    return result

@app.local_entrypoint()
def i_train(dataset='cifar10', epochs=1, push=True):
    import modal_integration.train_modal as tm
    cfg = {'epochs': epochs, 'use_wandb': True, 'wandb_project': 'ihrm-benchmarks', 'push_to_hub': push}
    result = tm.train_on_modal.remote(dataset=dataset, train_config=cfg, push_to_hub=push)
    print(result)
    return result

@app.local_entrypoint()
def i_eval(checkpoint, datasets='cifar10'):
    import modal_integration.train_modal as tm
    ds_list = [d.strip() for d in datasets.split(',')]
    result = tm.evaluate_on_modal.remote(checkpoint, ds_list)
    print(result)
    return result

@app.local_entrypoint()
def i_test():
    hello.remote()
if __name__ == '__main__':
    print('Unified app: ihrm')
    print('Entrypoints: run_download | run_train | run_eval | run_test')