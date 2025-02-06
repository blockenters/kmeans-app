import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)


def main():
    fontRegistered()
    plt.rc('font', family='NanumGothic')

    # 페이지 설정
    st.set_page_config(
        page_title="K-Means Clustering App",
        page_icon="📊",
        layout="wide"
    )

    # CSS 스타일 추가
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .info-box {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .step-header {
            font-weight: bold;
            margin-bottom: 1rem;
            color: #0066cc;
        }
        </style>
    """, unsafe_allow_html=True)

    # 헤더 섹션
    with st.container():
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title('📊 K-Means Clustering App!!')
        st.markdown('</div>', unsafe_allow_html=True)

    # 설명 섹션을 컬럼으로 분할
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 분석 도구 소개
        
        K-Means 클러스터링을 통해 데이터를 자동으로 그룹화하고 
        패턴을 발견하는 데이터 분석 도구입니다.
        """)
    
    with col2:
        st.markdown("""
        ### ✨ 주요 기능
        - 📁 자동 데이터 전처리
        - 🔄 자동 인코딩
        - 📈 WCSS 시각화
        - 📊 클러스터링 결과 제공
        """)

    # 구분선 추가
    st.markdown("---")

    # 파일 업로드 섹션
    st.markdown('<p class="step-header">STEP 1: 데이터 업로드</p>', unsafe_allow_html=True)
    file = st.file_uploader('CSV 파일을 업로드해주세요', type=['csv'])

    if file is not None:
        # 데이터 로드 섹션
        with st.container():
            st.markdown('<p class="step-header">STEP 2: 데이터 확인</p>', unsafe_allow_html=True)
            df = pd.read_csv(file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('##### 📋 데이터 미리보기')
                st.dataframe(df.head())
            
            with col2:
                st.markdown('##### ℹ️ 데이터 정보')
                st.markdown(f'- 전체 행 수: {df.shape[0]:,}개')
                st.markdown(f'- 전체 열 수: {df.shape[1]}개')
                st.markdown('- 결측치 현황:')
                st.dataframe(df.isna().sum())

        # 결측치 처리
        with st.container():
            st.info('🔍 결측치가 있는 행을 자동으로 제거합니다.')
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 컬럼 선택 섹션
        st.markdown('<p class="step-header">STEP 3: 분석할 컬럼 선택</p>', unsafe_allow_html=True)
        selected_columns = st.multiselect('분석에 사용할 컬럼을 선택해주세요', df.columns)

        if len(selected_columns) == 0 :
            return

        df_new = pd.DataFrame()
        # 4. 각 컬럼이, 문자열인지 숫자인지 확인. 
        for column in selected_columns:
            if is_integer_dtype(df[column]) :
                print(column + ' : int')
                print(df[column])
                df_new[column] = df[column]
                print(df_new[column])
            elif is_float_dtype(df[column]) :
                print(column + ' : float')
                df_new[column] = df[column]
            elif is_object_dtype(df[column]) :
                print(column + ' : object')
                print(f'이 컬럼의 유니크 갯수 : {df[column].nunique()}')

                if df[column].nunique() <= 2 :
                    # 레이블 인코딩
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else :
                    # 원핫 인코딩
                    ct = ColumnTransformer( [('encoder',OneHotEncoder(), [0])] , remainder='passthrough')
                    column_names = sorted( df[column].unique() )
                    df_new[column_names] = ct.fit_transform(df[column].to_frame() )
            else :
                st.text(f'{column} 컬럼은 K-Means에 사용 불가하므로 제외하겠습니다.')

        st.info('K-Means를 수행하기 위한 데이터 프레임 입니다.')
        st.dataframe(df_new)  

        st.subheader('최적의 k값을 찾기 위해 WCSS를 구합니다.')  

        
        # 데이터의 갯수가 클러스터링 갯수보다는 크거나 같아야 하므로
        # 해당 데이터의 갯수로 최대 k값을 정한다.
        st.text(f'데이터의 갯수는 {df_new.shape[0]}개 입니다.')
        if df_new.shape[0] < 10 :
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value= 2, max_value= df_new.shape[0])
            
        else :
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value= 2, max_value= 10)
            
        
        
        wcss = []
        for k in range(1, max_k+1) :
            kmeans = KMeans(n_clusters= k, random_state= 4)
            kmeans.fit(df_new)
            wcss.append( kmeans.inertia_ )

        # WCSS 시각화 섹션
        st.markdown('<p class="step-header">STEP 4: 최적의 군집 수 결정</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig1 = plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_k+1), wcss, marker='o')
            plt.title('The Elbow Method')
            plt.xlabel('클러스터 갯수')
            plt.ylabel('WCSS 값')
            st.pyplot(fig1)
        
        with col2:
            st.markdown("""
            #### 엘보우 방법이란?
            그래프가 꺾이는 지점(엘보우)이 
            최적의 군집 수로 간주됩니다.
            """)
            k = st.number_input('군집 수 선택', min_value=2, max_value=max_k)

        kmeans = KMeans(n_clusters= k, random_state= 4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장 되었습니다.')
        st.dataframe( df )
            
        

if __name__ == '__main__':
    main()

