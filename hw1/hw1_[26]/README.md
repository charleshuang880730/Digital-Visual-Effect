# VFX2023-Group26-HW1

## Main Program
### 執行完附上的py檔中的所有程式碼(如下)，即可獲得一張 tone mapping 過後的結果(tonemap.png)，以及獲得一張HDR影像(hdr.hdr)，此外還有RGB三個頻道的radiance map。
```shell
python hw1_group26.py
```


## Change dataset
### 若需要改使用其他Dataset，則可以修改py中有註解Main部分的程式碼，將讀檔路徑修改即可完成。此外由於部分註解使用中文，因此需使用UTF-8編碼開啟。
```shell
Aligned_Output_dir = 'Aligned' #<-若切換Dataset，請重新命名此檔名
PathToData = os.path.join('..','data','ImagesB','*jpg') # <--調整路徑可調整結果 {ImagesB,ImagesC}
path = sorted(glob.glob(PathToData))
ref_img_path = os.path.join("..","data","ImagesB","P1011524.jpg") # <--調整路徑可調整結果，設定為所選Data中任意一張影像，我們隨機挑了兩張做為reference {./ImagesB/P1011524.JPG, ./ImagesC/P1011533.JPG}請對應至所選的Data
```
