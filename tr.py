#%%
x = [4,2,3]
#%%
x.insert(0,123)
# %%
print(x)
# %%
indexed_rmse = list(enumerate(x))

# Sort the list of indexed RMSE values by the RMSE (second element in each tuple)
sorted_rmse = sorted(indexed_rmse, key=lambda x: x[1])


sorted_rmse

# %%
sorted_values = [rmse for _, rmse in sorted_rmse]
original_indexes = [index for index, _ in sorted_rmse]

# %%
sorted_values
# %%
original_indexes
# %%
