import numpy as np
import matplotlib.pyplot as plt

RMSE_train = np.array([0.78285579, 0.78286786, 
    0.78228329, 0.78318068, 0.78362326])

RMSE_test = np.array([0.87450032, 0.87789722, 
    0.87837809, 0.87688674, 0.87567991])

MAE_train = np.array([0.61569152, 0.61595894, 
    0.61565918, 0.61615892, 0.61624128])

MAE_test = np.array([0.68480832, 0.68693823, 
    0.6871088 , 0.68628395, 0.68543317])

plt.figure(1)
plt.subplot(211)
plt.plot(RMSE_train)
plt.plot(RMSE_test)
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
plt.plot(MAE_train)
plt.plot(MAE_test)
plt.legend(['train', 'test'], loc='upper left')
plt.show()