Python 版本:3.11.4
Pytorch版本:2.5.0+cu124

用python 3.11版來建立虛擬環境
安裝GPU驅動程式(最新版本)
下載PyTorch 指令:
pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
再安裝剩餘套件：
pip install -r requirements.txt