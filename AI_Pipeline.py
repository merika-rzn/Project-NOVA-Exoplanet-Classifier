import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

path = "cumulative.csv"      
df = pd.read_csv(path)         
print("Loaded dataframe shape:", df.shape)

print(df['koi_disposition'].value_counts())


train_df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
train_df['label'] = train_df['koi_disposition'].map({'CONFIRMED':1, 'FALSE POSITIVE':0})

candidate_features = [
    'koi_period','koi_duration','koi_depth','koi_prad','koi_teq','koi_insol','koi_model_snr',
    'koi_steff','koi_srad','koi_slogg','koi_kepmag',
    'koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec'
]
features = [c for c in candidate_features if c in train_df.columns]
print("Using features:", features)

X = train_df[features].copy()   
y = train_df['label'].copy()   

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_estimators=289,
    max_depth=30,
    max_features="sqrt",
    min_samples_leaf=3,
    min_samples_split=3,
    n_jobs=-1
)
clf.fit(X_train, y_train)


y_proba = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)


candidates_df = df[df['koi_disposition']=='CANDIDATE'].copy()
cand_X = candidates_df[features].copy()
for c in cand_X.columns:
    cand_X[c] = pd.to_numeric(cand_X[c], errors='coerce')
cand_X = cand_X.fillna(X.median())

if cand_X.shape[1] == 0:
    candidates_df['prob_planet'] = 0.0
else:
    candidates_df['prob_planet'] = clf.predict_proba(cand_X)[:,1]

ranking = candidates_df.sort_values('prob_planet', ascending=False)
output_cols = [c for c in ['kepoi_name','koi_period','koi_prad','koi_teq','prob_planet'] if c in ranking.columns]
ranking[output_cols].to_csv("top_candidates.csv", index=False)
print("Saved candidates ranking to top_candidates.csv")


from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv( "PSCompPars_2023.09.06_16.02.03.csv")
selected = [
    'pl_rade' , 'pl_orbsmax' , 'pl_eqt' , 'st_mass' , 'st_rad' , 'st_teff'
]
data = data.dropna(subset=['pl_bmasse']).copy()
X = data[selected].copy()
Y = data['pl_bmasse'].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c] , errors= 'coerce')
Y = pd.to_numeric(Y ,errors='coerce')
X = X.fillna(X.median())
Y = Y.fillna(Y.median())
Y_log = np.log1p(Y)
X_train , X_test , Y_train ,Y_test = train_test_split(X , Y_log , test_size= 0.18)
model = RandomForestRegressor(n_estimators=500 , max_depth= 20, random_state= 42, n_jobs= -1)
model.fit(X_train , Y_train)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)


artifacts = {
    "classifier": clf,          
    "regressor": model,          
    "features_clf": features,   
    "features_reg": selected    
}

# ذخیره یک فایل
joblib.dump(artifacts, "planet_pipeline.joblib")
