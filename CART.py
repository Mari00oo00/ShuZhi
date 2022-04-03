import pandas as pd
from sklearn.model_selection import train_test_split
import plot_tree
from cart_tree import CartTree as Cart

data = pd.read_csv("melon.txt", encoding='gbk')  # 格式为DataFrame
data2 = pd.read_csv("lenses.txt", sep="  ", engine='python')  # 这个数据用两个空格作为分隔符

del data2['id']
feat_name_lenses = data2.columns.tolist()  # 提取列索引作为特征集
del feat_name_lenses[-1]
data2.columns = range(data2.shape[1])
data2.iloc[0:] = data2.astype(float)
data2_train, data2_test = train_test_split(data2, test_size=0.4)

del data['编号']
del data['含糖率']
del data['密度']
feature_name = data.columns.tolist()  # 提取列索引作为特征集
del feature_name[-1]  # 删掉类别列
data.columns = range(data.shape[1])  # 重设列索引为数字形式
data_train, data_test = train_test_split(data, test_size=0.4)  # 用留出法划分训练集和验证集
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)


tree1 = Cart.cart_decision_tree(data_train, data_test, feature_name)
tree1_pre_division = Cart.tree_cart_classification(data_train, feature_name)
print('real:', list(data_test.iloc[:, -1]))
print(Cart.get_predict(tree1, data_test, feature_name))
tree2 = Cart.cart_decision_tree(data2_train, data2_test, feat_name_lenses, tree_type='regression')
tree2_pre_division = Cart.tree_regression(data2_train, feat_name_lenses)
print(tree2_pre_division)
plot_tree.createPlot(tree1_pre_division)
plot_tree.createPlot(tree1)
plot_tree.createPlot(tree2_pre_division)
plot_tree.createPlot(tree2)
