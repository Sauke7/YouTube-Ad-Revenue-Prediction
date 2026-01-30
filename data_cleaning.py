import pandas as pd
import numpy as np
# ==========================================
# Reading the csv file
# ==========================================

original_df = pd.read_csv('./youtube_ad_revenue_dataset.csv')
df = pd.read_csv('./youtube_ad_revenue_dataset.csv')
print(df.info())
print(df.describe())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 122400 entries, 0 to 122399
# Data columns (total 12 columns):
#  #   Column                Non-Null Count   Dtype  
# ---  ------                --------------   -----  
#  0   video_id              122400 non-null  object 
#  1   date                  122400 non-null  object 
#  2   views                 122400 non-null  int64  
#  3   likes                 116283 non-null  float64
#  4   comments              116288 non-null  float64
#  5   watch_time_minutes    116295 non-null  float64
#  6   video_length_minutes  122400 non-null  float64
#  7   subscribers           122400 non-null  int64  
#  8   category              122400 non-null  object 
#  9   device                122400 non-null  object 
#  10  country               122400 non-null  object 
#  11  ad_revenue_usd        122400 non-null  float64
# dtypes: float64(5), int64(2), object(5)
# memory usage: 11.2+ MB
#          views	      likes	          comments	    watch_time_minutes	 video_length_minutes	subscribers	    ad_revenue_usd
# count	122400.000000	116283.000000	116288.000000	116295.000000	        122400.000000	     122400.000000	 122400.000000
# mean	9999.856283	    1099.633618	    274.396636	    37543.827721	        16.014165	         502191.719902	 252.727210
# std	99.881260	    519.424089	    129.741739	    12987.724246	        8.083790	         288397.470103	 61.957052
# min	9521.000000	    195.000000	    48.000000	    14659.105562	        2.000142	         1005.000000	 126.590603
# 25%	9933.000000	    650.000000	    162.000000	    26366.320569	        9.004695	         252507.500000	 199.902018
# 50%	10000.000000	1103.000000	    274.000000	    37531.990337	        16.005906	         503465.500000	 252.749699
# 75%	10067.000000	1547.000000	    387.000000	    48777.782090	        23.021260	         752192.000000	 305.597518
# max	10468.000000	2061.000000	    515.000000	    61557.670089	        29.999799	         999997.000000	 382.768254


# ==========================================
#  HEADER CLEANING
# ==========================================

print(df.columns.tolist())
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("FIX APPLIED")
print(df.columns.tolist())

# ['video_id', 'date', 'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 
#  'subscribers', 'category', 'device', 'country', 'ad_revenue_usd']
# FIX APPLIED
# ['video_id', 'date', 'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 
#  'subscribers', 'category', 'device', 'country', 'ad_revenue_usd']

# ==========================================
#  CATEGORICAL TYPOS (FUZZY LOGIC)
# ==========================================

print(df['category'].unique())
print(df['device'].unique())
print(df['country'].unique())
cleanup_map = {
    'IN': 'IND',
    'DE' : 'DEU',
    'CA' : 'CAN',
    'US' : 'USA',
    'AU' : 'AUS',
    'UK' : 'UK',
}
df['country'] = df['country'].replace(cleanup_map)

# ['Entertainment' 'Gaming' 'Education' 'Music' 'Tech' 'Lifestyle']
# ['TV' 'Tablet' 'Mobile' 'Desktop']
# ['IND' 'CAN' 'UK' 'USA' 'DEU' 'AUS']

# ==========================================
#  ROW CLEANING
# ==========================================

df.category = df.category.str.strip()
df.device = df.device.str.strip()
df.country = df.country.str.strip()

# ==========================================
#  DROPPING DUPLICATE VALUES
# ==========================================

df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(df.info())

# np.int64(2400)
# Data columns (total 12 columns):
#  #   Column                Non-Null Count   Dtype  
# ---  ------                --------------   -----  
#  0   video_id              120000 non-null  object 
#  1   date                  120000 non-null  object 
#  2   views                 120000 non-null  int64  
#  3   likes                 114000 non-null  float64
#  4   comments              114000 non-null  float64
#  5   watch_time_minutes    114000 non-null  float64
#  6   video_length_minutes  120000 non-null  float64
#  7   subscribers           120000 non-null  int64  
#  8   category              120000 non-null  object 
#  9   device                120000 non-null  object 
#  10  country               120000 non-null  object 
#  11  ad_revenue_usd        120000 non-null  float64
# dtypes: float64(5), int64(2), object(5) 

# ==========================================
#  CHANGING DATE DATATYPE
# ==========================================

df['date'] = pd.to_datetime(df['date'])
print(df.info())

# BEFOR
#  1   date                  120000 non-null  object 
# AFTER
#  1   date                  120000 non-null  datetime64[ns]

# ==========================================
#  HANDLING NULL VALUES
# ==========================================

print(df["likes"].isnull().sum())
print(df["comments"].isnull().sum())
print(df["watch_time_minutes"].isnull().sum())

# np.int64(6000)
# np.int64(6000)
# np.int64(6000)

df["likes"].fillna(df["likes"].mean(),inplace=True)
df["comments"].fillna(df["comments"].mean(),inplace=True)
df["watch_time_minutes"].fillna(df["watch_time_minutes"].mean(),inplace=True)
print(df.info())

# Data columns (total 13 columns):
#  #   Column                Non-Null Count   Dtype         
# ---  ------                --------------   -----         
#  0   video_id              120000 non-null  object        
#  1   date                  120000 non-null  datetime64[ns]
#  2   views                 120000 non-null  int64         
#  3   likes                 120000 non-null  float64       
#  4   comments              120000 non-null  float64       
#  5   watch_time_minutes    120000 non-null  float64       
#  6   video_length_minutes  120000 non-null  float64       
#  7   subscribers           120000 non-null  int64         
#  8   category              120000 non-null  object        
#  9   device                120000 non-null  object        
#  10  country               120000 non-null  object        
#  11  ad_revenue_usd        120000 non-null  float64       
#  12  engagement_rate       120000 non-null  float64       
# dtypes: datetime64[ns](1), float64(6), int64(2), object(4)

# ==========================================
#  Outlier detection using IQR
# ==========================================

def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

IQR_likes_low, IQR_likes_high = iqr_bounds(df["likes"])
IQR_likes_low = max(0, IQR_likes_low)
print("likes =",IQR_likes_low, IQR_likes_high)
df["likes"] = df["likes"].clip(IQR_likes_low, IQR_likes_high)

# likes = 0 , 2800.5

IQR_comments_low, IQR_comments_high = iqr_bounds(df["comments"])
IQR_comments_low = max(0, IQR_comments_low)
print("comments =",IQR_comments_low, IQR_comments_high)
df["comments"] = df["comments"].clip(IQR_comments_low, IQR_comments_high)

# comments = 0 , 700.5

IQR_watch_time_minutes_low, IQR_watch_time_minutes_high = iqr_bounds(df["watch_time_minutes"])
IQR_watch_time_minutes_low = max(0, IQR_watch_time_minutes_low)
print("watch_time_minutes =",IQR_watch_time_minutes_low, IQR_watch_time_minutes_high)
df["watch_time_minutes"] = df["watch_time_minutes"].clip(IQR_watch_time_minutes_low, IQR_watch_time_minutes_high)

# watch_time_minutes = 0 , 80099.8291551471

# ==========================================
#  Feature Engineering
# ==========================================

df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]
print(df.info())

# Data columns (total 13 columns):
#  #   Column                Non-Null Count   Dtype         
# ---  ------                --------------   -----         
#  0   video_id              120000 non-null  object        
#  1   date                  120000 non-null  datetime64[ns]
#  2   views                 120000 non-null  int64         
#  3   likes                 120000 non-null  float64       
#  4   comments              120000 non-null  float64       
#  5   watch_time_minutes    120000 non-null  float64       
#  6   video_length_minutes  120000 non-null  float64       
#  7   subscribers           120000 non-null  int64         
#  8   category              120000 non-null  object        
#  9   device                120000 non-null  object        
#  10  country               120000 non-null  object        
#  11  ad_revenue_usd        120000 non-null  float64       
#  12  engagement_rate       120000 non-null  float64       
# dtypes: datetime64[ns](1), float64(6), int64(2), object(4)

# ==========================================
#  SKEWNESS iN DATA
# ==========================================

print(df["likes"].skew())
print(df["comments"].skew())
print(df["watch_time_minutes"].skew())

# np.float64(-0.0022019491413401196)
# np.float64(0.008230197889949997)
# np.float64(0.001530715691724734)

print(df.describe())

#                                 date          views          likes  ...    subscribers  ad_revenue_usd  engagement_rate
# count                         120000  120000.000000  120000.000000  ...  120000.000000   120000.000000    120000.000000
# mean   2024-12-08 03:24:11.233198848    9999.832333    1099.585044  ...  502291.970050      252.711361         0.137400
# min       2024-06-09 10:50:40.993199    9521.000000     195.000000  ...    1005.000000      126.590603         0.025492
# 25%    2024-09-07 10:50:40.993199104    9933.000000     673.000000  ...  252641.500000      199.892158         0.094965
# 50%    2024-12-08 10:50:40.993199104   10000.000000    1099.585044  ...  503633.500000      252.678607         0.137444
# 75%    2025-03-09 10:50:40.993199104   10067.000000    1524.000000  ...  752386.250000      305.613497         0.179922
# max       2025-06-08 10:50:40.993199   10468.000000    2061.000000  ...  999997.000000      382.768254         0.249554
# std                              NaN      99.918405     506.371911  ...  288364.967705       61.954125         0.052160

df.to_csv("cleaned_data.csv", index=False)