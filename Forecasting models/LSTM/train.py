#%%
import numpy as np
from properscoring import crps_ensemble
from main import *
def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_mae(var, var_hat):
    return np.sum(np.abs(var - var_hat)) / var.shape[0]

# def crps_ensemble(ensemble, observation):
#     n = len(ensemble)
#     sorted_ensemble = np.sort(ensemble)
#     cdf_values = np.arange(1, n + 1) / n
#
#     crps_value = 0.0
#     for i in range(n):
#         if sorted_ensemble[i] < observation:
#             crps_value += (cdf_values[i] - 1) ** 2
#         else:
#             crps_value += cdf_values[i] ** 2
#     # crps_value /= n
#     return crps_value

data = np.load('obs_seq.npy')
train_ratio = 0.8
train_idx = np.random.rand(data.shape[0]) < train_ratio
val_idx = ~train_idx
train_dataset = Bus_data(data[train_idx])
val_dataset = Bus_data(data[val_idx])

# Todo: test_dataset = ....

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
#%%
model = DeepAR(input_size=2, hidden_size=64, num_layers=3, dropout=0.1, seq_len=31)

# If you wan to Train the model
model = train_model(train_loader, val_loader, model, n_epochs=100, patience=10, lr=0.005)

# If you wan to Load the last model
# model.load_state_dict(torch.load(f'deepar.pth'))

#%% Forecast
mean_dat = np.load('mean.npy')
std_dat = np.load('std.npy')
mean_vec = mean_dat.reshape((31,2),order='F')
std_vec = std_dat.reshape((31,2),order='F')
test_dat = np.load('obs_seq_test.npy')
test_dat2 = test_dat.reshape((test_dat.shape[0],31,2),order='F')
true_dat = test_dat2*std_vec+mean_vec
test = Bus_data(test_dat)
index_test = np.load('obs_index.npy')

true_list_t = []
forecast_list_t = []
true_list_o = []
forecast_list_o = []
crps_list_t = []
crps_list_o = []
true_list_tt = []
forecast_list_tt = []
crps_list_tt = []

for jh in range(index_test.shape[0]):
    bus_num = index_test[jh,0]
    link_num = index_test[jh, 1]
    x, y = test[bus_num]  # the sample Id to test
    start_link = link_num  # The number of observed links
    total_link = y.shape[0]
    kk = model.forecast_samples(x[0][:start_link].unsqueeze(0),  # The observed sequence
                                x[1].unsqueeze(0),  # The features, the ID of all links
                                n=100)

    forecasts = kk.detach().numpy()
    reverse = forecasts*std_vec[start_link:,:]+mean_vec[start_link:,:]

    #%%
    # mean_forecats = kk.mean(dim=0).detach().numpy()
    # cov_forecasts = kk.var(dim=0).detach().numpy()

    mean_forecasts=reverse.mean(axis=0)
    forecast_list_t.append(mean_forecasts[:,0])
    forecast_list_o.append(mean_forecasts[:,1])
    true_list_t.append(true_dat[bus_num,start_link:,0])
    true_list_o.append(true_dat[bus_num,start_link:,1])
    true_list_tt.append(true_dat[bus_num,start_link:,0].sum())
    forecast_list_tt.append(mean_forecasts[:,0].sum())
    reverse_tt = reverse[:,:,0].sum(axis=1)
    crps_list_tt.append(crps_ensemble(mean_forecasts[:,0].sum(),reverse_tt))

    for lk in range(31-start_link):
        crps_list_t.append(crps_ensemble(true_dat[bus_num, lk+start_link, 0],reverse[:, lk, 0]))
        crps_list_o.append(crps_ensemble(true_dat[bus_num, lk + start_link, 1],reverse[:, lk, 1]))

# fig, ax = plt.subplots()
# ax.plot(y.detach().numpy()[:,0])
# ax.plot(range(start_link-1, total_link), mean_forecats[:,0])

t_true = np.hstack(true_list_t)
t_4cast = np.hstack(forecast_list_t)
o_true = np.hstack(true_list_o)
o_4cast = np.hstack(forecast_list_o)
tt_true = np.array(true_list_tt)
tt_4cast = np.array(forecast_list_tt)

t_mae = compute_mae(t_true,t_4cast)
o_mae = compute_mae(o_true,o_4cast)
t_rmse = compute_rmse(t_true,t_4cast)
o_rmse = compute_rmse(o_true,o_4cast)
t_crps = np.array(crps_list_t).mean()
o_crps = np.array(crps_list_o).mean()
tt_mae = compute_mae(tt_true,tt_4cast)
tt_rmse = compute_rmse(tt_true,tt_4cast)
tt_crps = np.array(crps_list_tt).mean()

print('link travel time:')
print('rmse, mae, crps')
print(t_rmse,t_mae,t_crps)
print('passenger occupancy:')
print('rmse, mae, crps')
print(o_rmse,o_mae,o_crps)
print('trip travel time:')
print('rmse, mae, crps')
print(tt_rmse,tt_mae,tt_crps)

