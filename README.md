# 數據集介紹

![image](https://github.com/11024244/mid/blob/main/jpg/01.png)

數據集來自**Kaggle**，品質很高，由知名醫院的專業人員嚴格審核標註，如圖所示數據有4種類別：

 •**CNV**：具有新生血管膜和相關視網膜下液的脈絡膜新血管形成
 
 •**DME**：糖尿病性黃斑水腫與視網膜增厚相關的視網膜內液
 
 •**DRUSEN**：早期AMD中存在多個玻璃疣
 
 •**NORMAL**：視網膜正常，沒有任何視網膜液或水腫
 
![image](https://github.com/11024244/mid/blob/main/jpg/02.png)

檔大小約為5GB，8萬多張圖像，分為訓練，測試，驗證三個資料夾，每個資料夾按照種類不同分成4個子資料夾，其次是具體圖像檔。

# 數據集下載

**掛載資料夾**：
    from google.colab import drive
    drive.mount('/content/gdrive/')
