# kmeans-app


## 한글 폰트 되도록 처리

나눔고딕 다운로드 : https://hangeul.naver.com/fonts/search?f=nanum 

custom_fonts 폴더 만들어서 ttf 파일 넣기

``` python
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)
```
``` python
def main():

    fontRegistered()
    plt.rc('font', family='NanumGothic')
```
    
