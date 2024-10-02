import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

from tqdm import tqdm
# from tqdm.notebook import tqdm  # jupyter で実行するとき


# リソースの選択（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# 乱数シード固定（再現性の担保）
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
fix_seed(seed)

# データローダーのサブプロセスの乱数のseedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Data preprocessing ----------------------------------------------------------
class LocalH5Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.h5')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.h5')])

        assert len(self.image_files) == len(self.mask_files), "画像ファイルとマスクファイルの数が一致しません"

        # 有効なデータのインデックスを保持するリスト
        self.valid_indices = []
        for i in range(len(self.image_files)):
            if self._is_valid_data(i):
                self.valid_indices.append(i)

    def _is_valid_data(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        with h5py.File(image_path, 'r') as image_file, h5py.File(mask_path, 'r') as mask_file:
            image = image_file['val_batch'][:]  # 'data'は画像データが格納されているキーと仮定
            mask = mask_file['mask_batch'][:]    # 'data'はマスクデータが格納されているキーと仮定

            # 50%以上が0の画素の画像を除外
            # if (image == 0).mean() > 0.5:
            #     return False

            # # 全て0のマスクを除外
            # if (mask == 0).all():
            #     return False

        return True

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # valid_indicesを使って有効なデータのファイル名を取得
        valid_index = self.valid_indices[index]
        image_path = os.path.join(self.image_dir, self.image_files[valid_index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[valid_index])

        with h5py.File(image_path, 'r') as image_file, h5py.File(mask_path, 'r') as mask_file:
            image = image_file['val_batch'][:]
            mask = mask_file['mask_batch'][:]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# データセットのルートディレクトリを指定
root = r"F:\test"

# imageフォルダとmaskフォルダのパス
image_dir = os.path.join(root, "img")
mask_dir = os.path.join(root, "mask")

# データセットの作成
train_dataset = LocalH5Dataset(image_dir, mask_dir) 

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")



# データローダーの作成
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=256,  # バッチサイズ
                                           shuffle=True,  # データシャッフル
                                           num_workers=2,  # 高速化
                                           pin_memory=True,  # 高速化
                                           worker_init_fn=worker_init_fn
                                           )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=2,
                                          pin_memory=True,
                                          worker_init_fn=worker_init_fn
                                          )


# Modeling --------------------------------------------------------------------

import torch
import torch.nn as nn

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        # 入力チャネル数6, 出力チャネル数16, カーネルサイズ7
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(6, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # 入力チャネル数16, 出力チャネル数64, カーネルサイズ3
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(256 * 256 * 64, 100)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(100, 256 * 256)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 256, 256)
        return x

# モデル・損失関数・最適化アルゴリスムの設定
model = Mymodel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)

# モデル訓練関数
def train_model(model, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader:
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label)
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


# 訓練の実行
epoch = 100
train_loss = []
test_loss = []

for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train_model(model, train_loader, test_loader)
    train_loss.append(train_l)
    test_loss.append(test_l)


# 学習状況（ロス）の確認
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()

# Evaluation ----------------------------------------------------------------

# 学習済みモデルから予測結果と正解値を取得
def retrieve_result(model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # Collect data
            preds.append(output)
            labels.append(label)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    # Returns as numpy (CPU環境の場合は不要)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return preds, labels


# 予測結果と正解値を取得
preds, labels = retrieve_result(model, test_loader)


# Other ----------------------------------------------------------------------

# 学習済みモデルの保存・ロード
path_saved_model = "./saved_model.pth"

# モデルの保存
torch.save(model.state_dict(), path_saved_model)
# モデルのロード
model = Mymodel()
model.load_state_dict(torch.load(path_saved_model))


# # Model summary
# from torchsummary import summary
# model = model().to(device)
# summary(model, input_size=(1, 256, 256))
