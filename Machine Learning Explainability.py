#Machine Learning Explainability
##Permutation Importance

#2
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

#4
perm2=PermutationImportance(second_model,random_state=1).fit(new_val_X,new_val_y)

eli5.show_weights(perm2,feature_names=new_val_X.columns.tolist())

##Partial Plots

#1
pdp_dist=pdp.pdp_isolate(model=first_model,dataset=val_X,model_features=base_features, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)

#2
inter1=pdp.pdp_interact(model=first_model,dataset=val_X,model_features=base_features, features=['pickup_longitude', 'dropoff_longitude'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['pickup_longitude', 'dropoff_longitude'], plot_type='contour')

plt.show()

#3
savings_from_shorter_trip = 15

#4
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

#6
y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2

#7
X1 = rand(n_samples)
X2 = rand(n_samples)
y = X1*X2

##SHAP Values

#1
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

#2
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


pdp_num_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature='number_inpatient')

pdp.pdp_plot(pdp_num_inpatient, 'number_inpatient')
plt.show()

#3
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


pdp_num_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature='time_in_hospital')

pdp.pdp_plot(pdp_num_inpatient, 'time_in_hospital')
plt.show()

#4
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()

#5
import shap


def patient_risk_factors(row):
    explainer = shap.TreeExplainer(my_model)
    shap_values = explainer.shap_values(row)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[0], shap_values[0], row)

patient_risk_factors(val_X.iloc[0])

##Advanced Uses of SHAP Values

#1
feature_with_bigger_range_of_effects = 'diag_1_428'

#2
bigger_effect_when_changed = 'diag_1_428'

#3
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)
shap.dependence_plot('num_medications', shap_values[1], small_val_X)