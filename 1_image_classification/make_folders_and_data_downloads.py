#!/usr/bin/env python
# coding: utf-8

# # 「第1章 画像分類」の準備ファイル
#
# - 本ファイルでは、第1章で使用するフォルダの作成とファイルのダウンロードを行います。
#

# In[1]:


import os
import urllib.request
import zipfile


# In[2]:


# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


# In[3]:


# ImageNetのclass_indexをダウンロードする
# Kerasで用意されているものです
# https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
save_path = os.path.join(data_dir, "imagenet_class_index.json")

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)


# In[4]:


# 1.3節で使用するアリとハチの画像データをダウンロードし解凍します
# PyTorchのチュートリアルで用意されているものです
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
save_path = os.path.join(data_dir, "hymenoptera_data.zip")

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
    print("if inner")
    # ZIPファイルを読み込み
    zip = zipfile.ZipFile(save_path)
    zip.extractall(data_dir)  # ZIPを解凍
    zip.close()  # ZIPファイルをクローズ

    # ZIPファイルを消去
    os.remove(save_path)


# In[5]:


#【※（実施済み）】

#ゴールデンリトリバーの画像を手動でダウンロード

#https://pixabay.com/ja/photos/goldenretriever-%E7%8A%AC-3724972/
#の640×426サイズの画像
#（画像権利情報：CC0 Creative Commons、商用利用無料、帰属表示は必要ありません）
#を、フォルダ「data」の直下に置く。
