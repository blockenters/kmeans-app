# K-Means Clustering 분석 앱

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green)

CSV 데이터를 업로드하여 K-Means 클러스터링 분석을 쉽게 수행할 수 있는 웹 애플리케이션입니다.

## 주요 기능

- CSV 파일 데이터 업로드 및 분석
- 자동 데이터 전처리 (결측치 처리)
- 문자형/숫자형 데이터 자동 인코딩
- WCSS(Within Cluster Sum of Squares) 시각화
- 최적의 클러스터 수(K) 결정을 위한 Elbow Method 제공
- 클러스터링 결과 데이터프레임 제공

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/kmeans-app.git
cd kmeans-app
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 한글 폰트 설정
- [네이버 나눔고딕](https://hangeul.naver.com/fonts) 다운로드
- 프로젝트 루트에 `custom_fonts` 폴더 생성
- 다운로드 받은 나눔고딕 폰트 파일(.ttf)을 `custom_fonts` 폴더에 복사

## 실행 방법

```bash
streamlit run app.py
```

## 사용 방법

1. 웹 브라우저에서 앱 접속
2. CSV 파일 업로드
3. 클러스터링에 사용할 컬럼 선택
4. WCSS 그래프를 통해 최적의 K값 확인
5. 원하는 클러스터 수 선택
6. 클러스터링 결과 확인

## 기술 스택

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib



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
    
