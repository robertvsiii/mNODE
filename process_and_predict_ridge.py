#!/usr/bin/env python
import pandas as pd
import numpy as np
from skbio.stats.composition import clr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from scipy import stats
import sys
PATH_metab_df = sys.argv[1] #### path to metabolomic profiles used in training
PATH_micro_df = sys.argv[2] #### path to microbiome profiles used in training
PATH_external_meta_df = sys.argv[3] #### path to metabolomic profiles used in testing/prediction
PATH_external_micro_df = sys.argv[4] #### path to metabolome profiles used in testing/prediction
PATH_metabolome_annotated = sys.argv[5] #### path to the annotation of metabolites in metabolomic profiles

## Load microbiome compositions and metabolomic profiles from the PRISM and NLIBD dataset
metab_df = pd.read_csv(PATH_metab_df, index_col=0)
micro_df = pd.read_csv(PATH_micro_df, index_col=0)

external_metab_df = pd.read_csv(PATH_external_meta_df, index_col=0)
external_micro_df = pd.read_csv(PATH_external_micro_df, index_col=0)

metabolome_annotated = pd.read_csv(PATH_metabolome_annotated, index_col=0)

samples = np.intersect1d(metab_df.columns.values, micro_df.columns.values)
num_samples = len(samples)

metab_df = metab_df[samples]
micro_df = micro_df[samples]

for c in micro_df.columns:
    micro_df[c] = pd.to_numeric(micro_df[c])
    
for c in metab_df.columns:
    metab_df[c] = pd.to_numeric(metab_df[c])
    
external_samples = np.intersect1d(external_metab_df.columns.values, external_micro_df.columns.values)
external_metab_df = external_metab_df[external_samples]
external_micro_df = external_micro_df[external_samples]

for c in external_micro_df.columns:
    external_micro_df[c] = pd.to_numeric(external_micro_df[c])

for c in external_metab_df.columns:
    external_metab_df[c] = pd.to_numeric(external_metab_df[c])
        
num_external_samples = len(external_samples)   


## Centered Log-Ratio DataFrames
metab_comp_df = pd.DataFrame(data=np.transpose(clr(metab_df.transpose() + 1)), 
                             index=metab_df.index, columns=metab_df.columns)

external_metab_comp_df = pd.DataFrame(data=np.transpose(clr(external_metab_df.transpose() + 1)), 
                                      index=external_metab_df.index, columns=external_metab_df.columns)
    

micro_comp_df = pd.DataFrame(data=np.transpose(clr(micro_df.transpose() + 1)), 
                             index=micro_df.index, columns=micro_df.columns)
external_micro_comp_df = pd.DataFrame(data=np.transpose(clr(external_micro_df.transpose() + 1)), 
                             index=external_micro_df.index, columns=external_micro_df.columns)

micro_comp_df = micro_comp_df.transpose()
metab_comp_df = metab_comp_df.transpose()
external_micro_comp_df = external_micro_comp_df.transpose()
external_metab_comp_df = external_metab_comp_df.transpose()
print(micro_comp_df.shape, metab_comp_df.shape, external_micro_comp_df.shape, external_metab_comp_df.shape)


## Use the PRISM as the training set and NLIBD as the test set
X_train = micro_comp_df.values
y_train = metab_comp_df.values
X_test = external_micro_comp_df.values
y_test = external_metab_comp_df.values


## Save the training and test data
np.savetxt("./processed_data/X_train.csv", X_train, delimiter=',')
np.savetxt("./processed_data/y_train.csv", y_train, delimiter=',')
np.savetxt("./processed_data/X_test.csv", X_test, delimiter=',')
np.savetxt("./processed_data/y_test.csv", y_test, delimiter=',')


## Save compound names
metabolome_raw = pd.read_csv(PATH_metab_df, index_col=0)
compound_names = metabolome_annotated.reindex(metabolome_raw.index)['Compound Name']
np.savetxt("./processed_data/compound_names.csv", compound_names.values, delimiter='\t', fmt = '%s')




##### Ridge Regression #####

## 5-fold cross-validation to find optimal alpha

def mean_rho_test(y_t,y_p):
    rhovec = []
    for k in range(len(y_p.T)):
        rho,p = stats.spearmanr(y_p[:,k],y_t[:,k])
        rhovec.append(rho)
    return np.asarray(rhovec).mean()
mean_rho_score = make_scorer(mean_rho_test)

alpha_grid = {'alpha':np.linspace(0.1,1,10)}
ridge = Ridge(alpha=.5,max_iter=100000,fit_intercept=True)
ridge_opt = GridSearchCV(ridge,alpha_grid,scoring=mean_rho_score,cv=5,n_jobs=-1,verbose=2)
out=ridge_opt.fit(X_train,y_train)

## Predict and score

Y_pred = pd.DataFrame(ridge_opt.predict(X_test))
rhovec = Y_pred.corrwith(pd.DataFrame(y_test),method='spearman').values
rhos_valid = rhovec[~compound_names.isna().values.squeeze()]
rhos_valid.sort()
rhovec.sort()
score_1 = rhovec.mean()
score_2 = rhovec[-50:].mean()
score_3 = (rhovec>0.5).sum()
score_3_valid = (rhos_valid>0.5).sum()

print('The mean Spearman C.C. for all metabolites: '+str(score_1))
print('Top 50 Spearman C.C.: '+str(score_2))
print('The number of metabolites with Spearman C.C. > 0.5: '+str(score_3))
print('The number of annotated metabolites with Spearman C.C. > 0.5: '+str(score_3_valid))