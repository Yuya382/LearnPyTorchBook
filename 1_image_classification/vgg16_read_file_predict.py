#!/usr/bin/env python
# coding: utf-8

# パッケージのimport
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

#自作class 呼び出す
import ILSVRCPredictor as ilsvr
import BaseTransform as base

# PyTorchのバージョン確認
#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)


# # VGG-16の学習済みモデルをロード

# 学習済みのVGG-16モデルをロード
# 初めて実行する際は、学習済みパラメータをダウンロードするため、実行に時間がかかります

class vgg16(object):

    def predict(self,image_file_path):
            # VGG-16モデルのインスタンスを生成
        use_pretrained = True  # 学習済みのパラメータを使用
        net = models.vgg16(pretrained=use_pretrained)
        net.eval()  # 推論モードに設定



        # 1. 画像読み込み
        img = Image.open(image_file_path)  # [高さ][幅][色RGB]

        # 3. 画像の前処理と処理済み画像の表示
        resize = 224
        mean = (0.485, 0.456, 0.406) #固定値
        std = (0.229, 0.224, 0.225)#固定値
        transform = base.BaseTransform(resize, mean, std)

        img_transformed = transform(img)  # torch.Size([3, 224, 224])

        # (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
        img_transformed = img_transformed.numpy().transpose((1, 2, 0))
        img_transformed = np.clip(img_transformed, 0, 1)

        # # 学習済みVGGモデルで手元の画像を予測
        # ILSVRCのラベル情報をロードし辞意書型変数を生成します
        ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))

        # ILSVRCPredictorのインスタンスを生成します
        predictor = ilsvr.ILSVRCPredictor(ILSVRC_class_index)

        # 入力画像を読み込む
        img = Image.open(image_file_path)  # [高さ][幅][色RGB]

        # 前処理の後、バッチサイズの次元を追加する
        img_transformed = transform(img)  # torch.Size([3, 224, 224])
        inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

        # モデルに入力し、モデル出力をラベルに変換する
        out = net(inputs)  # torch.Size([1, 1000])
        result = predictor.predict_max(out)

        # 予測結果を出力する
        print("入力画像の予測結果 :", result)
