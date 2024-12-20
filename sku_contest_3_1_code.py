#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


workload = pd.read_excel(r"C:\Users\USER\OneDrive\Desktop\sk_cont31\data\LOT 물량.xlsx")
operation = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\sk_cont31\data\PRODUCTION_TREND.csv", encoding='CP949')
ccm = pd.read_excel(r"C:\Users\USER\OneDrive\Desktop\sk_cont31\data\CCM 측정값.xlsx")


# In[4]:


workload.head()


# In[5]:


operation.head()


# In[6]:


ccm.head()


# In[7]:


operation = operation[['LOT_NO', 'WC_CD', 'RESOURCE_CD', 'INSRT_DT', 'SEQ_NO', 'CR_TEMP', 'TRD_TEMP_SP', 'TRD_TEMP_PV', 'TRD_SPEED1', 'TRD_SPEED2', 'TRD_SPEED3', 'TRD_SPEED4']]


# In[8]:


operation.columns = ['LOT번호', '공정코드', '설비번호', '공정일시', '공정진행시간', '목표온도', '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4']


# In[9]:


ccm =ccm[['lot_no', 'seq', 'oper_id', '염색 색차 DE']]


# In[10]:


ccm.columns =['LOT번호', '검사차수', '작업명', '염색색차 DE']


# In[11]:


workload = workload[['PRODT_ORDER_NO', 'JOB_CD', 'EXT1_QTY(투입중량 (KG))', 'EXT2_QTY (액량 (LITER))', '단위중량', '염색 가동 길이']]


# In[12]:


workload.columns = ['LOT번호', '공정코드', '투입중량(kg)', '투입액량(L)', '단위중량(kg)', '염색길이(m)']


# In[13]:


ccm.describe(include=object)


# In[14]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# In[15]:


# 한글폰트 사용을 위한 환경 설정. '맑은 고딕'으로 폰트 설정
path ='C:\\WINDOWS\\Fonts\\malgunbd.ttf'
font_name = fm.FontProperties(fname = path, size =10).get_name()
plt.rc('font', family = font_name)


# In[16]:


workload.hist(bins=20, figsize=(8,5), grid=False)
plt.tight_layout()


# In[92]:


workload['투입액량(L)'].unique()


# In[17]:


operation.hist(bins=20, figsize=(10,12), grid=False, layout=(4,2))
plt.tight_layout()


# In[18]:


ccm.hist(bins=20, figsize=(8,5), grid=False)


# In[19]:


ccm[ccm['LOT번호'].str.islower()]


# In[20]:


ccm['LOT번호'] = ccm['LOT번호'].str.capitalize()


# In[21]:


ccm['LOT번호'].str.islower().sum()


# In[22]:


# SET1 중복데이터 개수 확인
workload.duplicated(keep=False).sum()
#keep=False -> 모든 중복데이터 포함, keep='first' -> 첫번째 중복데이터는 제외


# In[23]:


# SET1 중복데이터 내용 확인
workload[workload.duplicated(keep=False)]


# In[24]:


# 중복데이터 한 개씩만 남기고 모두 삭제
workload2 = workload[~workload.duplicated(keep='first')]


# In[25]:


# 인덱스번호 리셋
workload2.reset_index(drop=True, inplace=True)
# 중복데이터 개수 재확인
workload2.duplicated().sum()


# In[26]:


# SET1 중복데이터 삭제 전후 데이터 크기 비교
print(workload.shape)
print(workload2.shape)


# In[27]:


# 중복데이터 개수 확인
operation.duplicated().sum()


# In[28]:


# LOT번호+공정진행시간 순으로 데이터 재정렬
operation.sort_values(by=['LOT번호', '공정진행시간'], ascending=True, inplace=True, ignore_index=True)
# 중복데이터 내용 확인
operation[operation.duplicated(keep=False)] 


# In[29]:


# 중복데이터 한 개씩만 남기고 모두 삭제
operation2 = operation[~operation.duplicated(keep='first')]
# 인덱스번호 리셋
operation2.reset_index(drop=True, inplace=True)
# 중복데이터 개수 재확인
operation2.duplicated().sum()


# In[30]:


ccm.duplicated().sum()


# In[31]:


#중복 데이터 확인
ccm[ccm.duplicated(keep=False)]


# In[32]:


# 중복데이터 한 개씩만 남기고 모두 삭제
ccm2 = ccm[~ccm.duplicated(keep='first')]
# 인덱스번호 리셋
ccm2.reset_index(drop=True, inplace=True)
# 중복데이터 개수 재확인
ccm2.duplicated().sum()


# In[33]:


# SET3 중복데이터 삭제 전후 데이터 크기 비교
print(ccm.shape)
print(ccm2.shape)


# In[34]:


# LOT번호순으로 데이터 재정렬
workload2.sort_values(by='LOT번호', ascending=True, ignore_index=True, inplace=True)


# In[35]:


# 데이터 유일성을 벗어나는 데이터 개수 확인
(workload2.groupby('LOT번호').size() >1).sum()


# In[36]:


# 비유일성 데이터 내용 확인
workload2[workload2.duplicated(subset='LOT번호', keep=False)]


# In[37]:


# 비유일성 데이터 삭제
workload3 = workload2.drop(1647, axis=0) #index=1647 위치의 행(row) 삭제
workload3 = workload3.drop(1939, axis=0) #index=1939 위치의 행(row) 삭제
# 인덱스번호 리셋
workload3.reset_index(drop=True, inplace=True)
# LOT번호 기준으로 데이터 유일성 재확인
(workload3.groupby('LOT번호').size() >1).sum()


# In[38]:


# LOT번호+공정진행시간 순으로 데이터 재정렬
operation2.sort_values(by=['LOT번호', '공정진행시간'], ascending=True, inplace=True, ignore_index=True)


# In[39]:


# 데이터 유일성을 벗어나는 데이터 개수 확인
(operation2.groupby(['LOT번호', '공정코드', '공정진행시간']).size() >1).sum()


# In[40]:


# 비유일성 데이터 내용 확인
operation2[operation2.duplicated(subset=['LOT번호', '공정코드', '공정진행시간'], keep=False)].head(10)


# In[41]:


# 중복데이터 한 개씩만 남기고 모두 삭제
operation3 = operation2[~operation2.duplicated(subset=['LOT번호', '공정코드', '공정진행시간'], keep='last')]
# 인덱스번호 리셋
operation3.reset_index(drop=True, inplace=True)
# 데이터 유일성 재확인
(operation3.groupby(['LOT번호', '공정코드', '공정진행시간']).size() >1).sum()


# In[42]:


# LOT번호+검사차수 순으로 데이터 재정렬
ccm2.sort_values(by=['LOT번호', '검사차수'], ascending=True, ignore_index=True, inplace=True)


# In[43]:


# 데이터 유일성을 벗어나는 데이터 개수 확인
(ccm2.groupby(['LOT번호', '검사차수']).size() >1).sum()


# In[44]:


# 비유일성 데이터 내용 확인
ccm2[ccm2.duplicated(subset=['LOT번호', '검사차수'], keep=False)].head(20)


# In[45]:


# LOT번호 기준으로 가장 마지막 값(=마지막 검사차수값)만 추출
ccm3 = ccm2.groupby(['LOT번호']).last()
# groupby를 통해 인덱스가 된 LOT번호를 다시 컬럼명으로 변경
ccm3.reset_index(drop=False, inplace=True)
# 추출 데이터 확인
ccm3


# In[46]:


# 마지막 검사차수 추출 전후 데이터 개수 비교
print(ccm.shape)
print(ccm3.shape)


# In[47]:


# 데이터 유일성 재확인
ccm3.duplicated(subset=['LOT번호', '검사차수']).sum()


# In[48]:


# 목표온도 대비 진행온도 계산 및 변수 생성
operation3['목표대비 진행온도'] = operation3['진행온도'] / operation3['목표온도']
# 지시온도 대비 진행온도 계산 및 변수 생성
operation3['지시대비 진행온도'] = operation3['진행온도'] / operation3['지시온도']


# In[49]:


operation3


# In[50]:


# inf -> 0으로 대체
operation3.replace([np.inf, -np.inf], 0, inplace=True)


# In[51]:


# NaN -> 0으로 대체
operation3.fillna(0, inplace=True)


# In[52]:


# 결과를 저장할 새 데이터프레임 생성
operation4 = pd.DataFrame() 
# 각 LOT번호 그룹을 룹핑하면서 공정진행시간% 계산
for name, group in operation3.groupby('LOT번호'):
    group.sort_values(by=['공정진행시간'], ascending=True, inplace=True, ignore_index=True)
 
 # 전체 공정시간 대비 진행시간(%)
    group['공정진행시간(%)'] = round(group['공정진행시간'] / group['공정진행시간'].max() *100, 2)
 
 # 결과 저장
    operation4 = pd.concat([operation4, group])
# 인덱스 리셋
operation4.reset_index(drop=True, inplace=True)


# In[53]:


operation4


# In[54]:


operation4 = operation4[['LOT번호', '공정코드', '설비번호', '공정진행시간(%)',
'목표온도', '지시온도', '진행온도', '목표대비 진행온도', '지시대비 진행온도', '포속1', '포속2', '포속3', '포속4']]


# In[55]:


operation4.head()


# In[56]:


# LOT 물량 데이터 + CCM 데이터 결합 - ‘LOT번호’ 기준
df = ccm3.merge(workload3, how='inner', on='LOT번호')
#inner: 양쪽에 동시에 존재하는 LOT번호 기준으로 결합


# In[57]:


df.head()


# In[58]:


# 설비데이터(시계열) + 그 외 데이터(비시계열) 결합 - 'LOT번호', '공정코드' 기준
df2 = df.merge(operation4, how='inner', on=['LOT번호', '공정코드'])


# In[59]:


df2.head()


# In[60]:


# 데이터 크기 변화 확인
print('SET1 데이터 크기: {}'.format(workload3.shape))
print('SET2 데이터 크기: {}'.format(operation4.shape))
print('SET3 데이터 크기: {}'.format(ccm3.shape))
print('\nSET1+SET3 데이터 크기: {}'.format(df.shape))
print('SET1+SET2+SET3 데이터 크기: {}'.format(df2.shape))


# In[61]:


# LOT번호 유일값 개수 변화 확인
print('SET1 LOT번호 유일값 개수: {}'.format(workload3['LOT번호'].nunique()))
print('SET2 LOT번호 유일값 개수: {}'.format(operation4['LOT번호'].nunique()))
print('SET3 LOT번호 유일값 개수: {}'.format(ccm3['LOT번호'].nunique()))
print('\nSET1+SET3 LOT번호 유일값 개수: {}'.format(df['LOT번호'].nunique()))
print('SET1+SET2+SET3 LOT번호 유일값 개수: {}'.format(df2['LOT번호'].nunique()))


# In[62]:


# 변수 순서 변경
df2 = df2[['LOT번호', '작업명', '공정코드', '설비번호',
'단위중량(kg)', '투입중량(kg)', '투입액량(L)', '염색길이(m)',
'공정진행시간(%)', '목표온도', '지시온도', '진행온도', '목표대비 진행온도', '지시대비 진행온도',
'포속1', '포속2', '포속3', '포속4', '염색색차 DE']]


# In[63]:


df2.head()


# In[64]:


df2.describe()


# In[65]:


# 이상치 개수 확인
len(df2[df2['단위중량(kg)'] ==0])


# In[66]:


# 해당 이상치에 속하는 LOT번호 확인
df2[df2['단위중량(kg)'] ==0]['LOT번호'].unique()


# In[67]:


# 단위중량 이상치에 해당하는 데이터 모두 제거
df3 = df2[df2['단위중량(kg)'] >0]
df3.reset_index(drop=True, inplace=True)


# In[68]:


# 데이터 크기 변화 확인
print('이상치 제거 전 데이터 크기: {}'.format(df2.shape))
print('이상치 제거 후 데이터 크기: {}'.format(df3.shape))


# In[69]:


# 이상치 개수 확인
len(df3[df3['염색길이(m)'] ==1])


# In[70]:


# 해당 이상치에 속하는 LOT번호 확인
df3[df3['염색길이(m)'] ==1]['LOT번호'].unique()


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt

# boxplot 그리기
plt.figure(figsize=(10, 6))
plt.boxplot(df3['염색길이(m)'])
plt.title('Boxplot of {column_name}')
plt.ylabel("column_name")
plt.grid(True)
plt.show()


# In[72]:


data= df3['염색길이(m)']
# 박스플롯의 주요 통계 정보 출력
quartiles = data.quantile([0.25, 0.5, 0.75])
q1 = quartiles[0.25]
median = quartiles[0.5]
q3 = quartiles[0.75]
iqr = q3 - q1
lower_whisker = max(data.min(), q1 - 1.5 * iqr)
upper_whisker = min(data.max(), q3 + 1.5 * iqr)

print("Statistics for {column_name}:")
print(f"Minimum: {data.min()}")
print(f"1st Quartile (Q1): {q1}")
print(f"Median: {median}")
print(f"3rd Quartile (Q3): {q3}")
print(f"Maximum: {data.max()}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Lower Whisker: {lower_whisker}")
print(f"Upper Whisker: {upper_whisker}")


# In[73]:


df3['LOT번호'].describe()


# In[74]:


# 각 수치형 변수의 unique 값 개수 확인
for col in df3.describe().columns:
    print( col,':', len(df3[col].value_counts()))


# In[75]:


# 각 범주형 변수의 unique 값 개수 확인
for col in df3.describe(include=object).columns:
    print( col,':', len(df3[col].value_counts()))


# In[76]:


# 불필요한 열 제거
df4 = df3.drop(['공정코드'], axis=1)


# In[77]:


# '작업명' 변수 레이블의 데이터 분포 확인
df4['작업명'].value_counts()


# In[78]:


# '설비번호' 변수 레이블의 데이터 분포 확인
df4['설비번호'].value_counts()


# In[79]:


# 필요 라이브러리 불러오기
import seaborn as sns
# 그래프 '-' 음수부호 오류 해결
plt.rcParams['axes.unicode_minus'] =False 
# 차트 폰트 설정
plt.rcParams['font.size'] =15
plt.rcParams['font.family'] ='Malgun Gothic'


# In[112]:


# 독립변수 정의
#X = df4[['단위중량(kg)', '투입중량(kg)', '투입액량(L)', '염색길이(m)', '공정진행시간(%)',
#'목표온도', '지시온도', '진행온도', '목표대비 진행온도', '지시대비 진행온도',
#'포속1', '포속2', '포속3', '포속4']]
# 피어슨 상관계수 계산
corr = X.corr()
# 히트맵 그리기
fig = plt.figure(figsize=(17,10))
sns.heatmap(corr, vmin=-1, vmax=1, cmap='BrBG', annot=True, fmt=".2f", linewidths=.5)
plt.show()


# In[1]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df4[['단위중량(kg)', '투입중량(kg)', '염색길이(m)', '공정진행시간(%)',
'목표온도', '지시온도', '목표대비 진행온도', '지시대비 진행온도',
'포속1', '포속3', '포속4']]
# vif 데이터프레임 생성
vif = pd.DataFrame()

# VIF 값 계산
vif['VIF_Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['Feature'] = X.columns

# VIF 값 오름차순으로 정렬
vif = vif.sort_values(by='VIF_Factor', ascending=True)

# 결과 출력
print(vif)


# In[81]:


# 독립변수 정의
Xcols = ['단위중량(kg)', '투입중량(kg)', '투입액량(L)', '염색길이(m)', '공정진행시간(%)',
'목표온도', '진행온도', '목표대비 진행온도', '지시대비 진행온도', '포속1', '포속3', '포속4']
# 종속변수 정의
ycol = ['염색색차 DE']
# 독립변수 데이터셋 추출
X = df4[Xcols]
# 종속변수 데이터셋 추출
y = df4[ycol]


# In[82]:


# 필요 패키지 불러오기
from sklearn.model_selection import train_test_split
# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[83]:


# 필요 패키지 불러오기
from sklearn.preprocessing import StandardScaler
# 데이터 표준화 스케일러 정의
scaler = StandardScaler()
# 훈련셋에 스케일러 적용 - fit, transform 모두 적용
X_train_sc = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_sc, index=X_train.index, columns=X_train.columns)
# 테스트셋에 스케일러 적용 - transform만 적용
X_test_sc = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_sc, index=X_test.index, columns=X_test.columns)


# In[84]:


X_train.head()


# In[85]:


# 필요 패키지 불러오기
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
# 하이퍼파라미터 정의
parameters = {"n_estimators": [20, 50],
 "learning_rate": [0.1, 0.15],
 "max_depth": [2, 4],
 "objective":['reg:squarederror']}
# K FOLD 교차검증
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# 모델 생성
model_xgb = GridSearchCV(XGBRegressor(random_state=42), parameters, cv=kfold, verbose=2, n_jobs=-1)


# In[86]:


# 모델 학습
model_xgb.fit(X_train, y_train)
print(f"XGBoost 회귀모델 best 파라미터: {model_xgb.best_params_}")
# 모델 최적화
best_model_xgb = model_xgb.best_estimator_
best_model_xgb.fit(X_train, y_train)


# In[87]:


# 모델 예측
y_pred_xgb = best_model_xgb.predict(X_test)
# 예측 결과 확인
y_pred_xgb


# In[88]:


# 필요 패키지 불러오기
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
# 모델 성능 계산
r2 = r2_score(y_test, y_pred_xgb)
adj_r2 =1 - (1 - r2) * (len(y_test) -1) / (len(y_test) - X_test.shape[1] -1)
rmse = sqrt(MSE(y_test, y_pred_xgb))
mae = mean_absolute_error(y_test, y_pred_xgb)
# 결과 인쇄
print(f"Adjusted. R2_score : {round(adj_r2,3)}")
print(f"RMSE score : {round(rmse,3)}")
print(f"MAE score : {round(mae,3)}")


# In[89]:


# 실제값과 예측값 비교 플롯 그리기
plt.title(f'Prediction vs Actual (XGBoost)')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.grid()
plt.scatter(y_pred_xgb, y_test)


# In[90]:


# 필요 라이브러리 불러오기
from xgboost import plot_importance
# 특성 중요도 그리기
fig, ax = plt.subplots(figsize=(10, 9))
plot_importance(best_model_xgb, height=0.8, grid=False, ax=ax)
plt.xlabel('특성 중요도')
plt.ylabel('변수명')
plt.show()

