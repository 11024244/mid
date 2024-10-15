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

```py
from google.colab import drive

drive.mount('/content/gdrive/')
```
按照提示進行驗證，結果如下：

![image](https://github.com/11024244/mid/blob/main/jpg/03.png)

**kaggle資料下載**：

創建**kaggle**帳戶並下載**kaggle.json**檔。 創建帳戶這裡就不介紹了，創建完帳戶後在“我的帳戶”-“API”中選擇“CREATE NEW API TOKEN”，然後下載**kaggle.json**檔。

**建立kaggle資料夾**：
```py
!mkdir -p ~/.kaggle
```
**將kaggle.json資料夾複製到指定資料夾**：
```py
!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
```
**測試是否成功**：
```py
!kaggle competitions list
```
![image](https://github.com/11024244/mid/blob/main/jpg/04.png)

**下載資料集**：
```py
!kaggle datasets download -d paultimothymooney/kermany2018
```
![image](https://github.com/11024244/mid/blob/main/jpg/05.png)

**將文件解壓至google雲盤**：
```py
!unzip "/content/OCT2017.zip" -d "/content/gdrive/My Drive"
```
