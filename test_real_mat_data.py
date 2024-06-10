import numpy as np
import scipy.io
import tensorflow as tf

# 从MATLAB文件中读取变量
mat = scipy.io.loadmat('push2.mat')
wifiVelocityInterp = mat['wifiVelocityInterp'].flatten()
acouVelocityInterp = mat['acouVelocityInterp'].flatten()
initial_position_x = mat['initial_position_x'][0, 0]
initial_position_y = mat['initial_position_y'][0, 0]

original_x = np.linspace(0, 1, len(wifiVelocityInterp))

# Define the new x-coordinates for interpolation
new_x = np.linspace(0, 1, 120)

# Interpolate the wifiVelocityInterp array
wifi_interp_func = scipy.interpolate.interp1d(original_x, wifiVelocityInterp, kind='linear')
wifiVelocityInterp_new = wifi_interp_func(new_x)

# Interpolate the acouVelocityInterp array
acou_interp_func = scipy.interpolate.interp1d(original_x, acouVelocityInterp, kind='linear')
acouVelocityInterp_new = acou_interp_func(new_x)

# Now wifiVelocityInterp_new and acouVelocityInterp_new have length 200
# print("Interpolated wifiVelocityInterp:", wifiVelocityInterp_new)
# print("Interpolated acouVelocityInterp:", acouVelocityInterp_new)

# 获取列数
num_cols = wifiVelocityInterp_new.shape[0]

# 构建三维np.array
test_real_data = np.zeros((1, num_cols, 4))
test_real_data[0, :, 0] = wifiVelocityInterp_new
test_real_data[0, :, 1] = acouVelocityInterp_new
test_real_data[0, :, 2] = np.full((num_cols,), initial_position_x)
test_real_data[0, :, 3] = np.full((num_cols,), initial_position_y)

print(test_real_data)

custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

# 加载模型时使用 custom_objects 参数
loaded_model = tf.keras.models.load_model('./saved_models/model_checkpoint_epoch_03_loss_0.94.keras', custom_objects=custom_objects)
# loaded_model = tf.keras.models.load_model('./saved_models/model_checkpoint_epoch_10_loss_0.01.keras', custom_objects=custom_objects)



pred = loaded_model.predict(test_real_data)

# Optionally, visualize the results
import matplotlib.pyplot as plt
# start = 5
print(test_real_data.shape)
for i in range(test_real_data.shape[0]):  # Plot the first 5 trajectories
    plt.figure(figsize=(6, 5))
    plt.plot(pred[i, :, 0], pred[i, :, 1], label='Predicted Trajectory')
    # plt.plot(real_test[i, :, 0], real_test[i, :, 1], label='Actual Trajectory')
    # print(X_test[i, :, 2], X_test[i, :, 3])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Trajectory {i+1}')
    plt.legend()
    plt.show()