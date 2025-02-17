conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
conda install wandb omegaconf numpy pandas scikit-learn tqdm biopython mango htmd pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 pyg -c pytorch -c nvidia -c conda-forge -c bioconda -c omnia  -c pyg


conda install pytorch torchvision torchaudio pytorch-cuda pyg -c pytorch -c nvidia -c conda-forge  -c pyg
conda install wandb omegaconf numpy scikit-learn tqdm  matplotlib htmd  -c acellera -c conda-forge
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg
pip install arm-mango