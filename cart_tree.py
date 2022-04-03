import numpy as np


class CartTree:
    @staticmethod
    def majority_category(category):
        """
        :param category: 类别列表(格式要求为Series)
        :return: 含样本数最多的类的值
        """
        category = list(category)
        best_category = category[0]
        num = category.count(best_category)
        for value in category:
            if num < category.count(value):
                best_category = value
                num = category.count(value)
        return best_category

    @staticmethod
    def split_data_for_cart(data_set, loc_feature, value):  # 划分出左右两个数据集
        """
        :param data_set: 数据集
        :param loc_feature: 特征所在的列的位置
        :param value: 特征的某一个取值
        :return: D1和D2两个数据集
        """
        data_set = data_set.reset_index(drop=True)
        data_set.columns = range(data_set.shape[1])
        # 重新排一下索引
        if type(value) != int and type(value) != float:  # 非整型形式
            left_data = data_set[data_set[loc_feature] == value]  # 左结点的数据集
            right_data = data_set[data_set[loc_feature] != value]  # 右结点数据集
        else:  # 实数形式
            left_data = data_set[data_set[loc_feature] <= value]  # 左结点的数据集
            right_data = data_set[data_set[loc_feature] > value]  # 右结点数据集
        # 重新设置数据集索引
        del left_data[loc_feature]
        left_data = left_data.reset_index(drop=True)
        left_data.columns = range(left_data.shape[1])
        del right_data[loc_feature]
        right_data = right_data.reset_index(drop=True)
        right_data.columns = range(right_data.shape[1])

        return left_data, right_data

    @staticmethod
    def cal_leaves_number(tree, num):
        """
        :param tree: 子树
        :param num:当前结点的叶结点数
        :return: 递归后最终返回该子树的叶结点数
        """
        node_feature = list(tree.keys())[0]  # 提取根结点
        for value in list(tree[node_feature].keys()):
            if value == 'PreDivision':  # 遇到未划分的标签就跳过
                continue
            elif type(tree[node_feature][value]) == dict:  # 是字典形式说明还是子树，进入递归
                num = CartTree.cal_leaves_number(tree[node_feature][value], num)
            elif type(tree[node_feature][value]) != dict:
                num += 1
        return num

    @staticmethod
    def calculate_gini(data_set):
        """
        :param data_set: 数据集（划分后的）
        :return:基尼指数值
        """
        data_set = data_set.reset_index(drop=True)
        data_set.columns = range(data_set.shape[1])
        # 重新排一下索引
        category_list = data_set.iloc[:, -1]  # 提取种类列表
        category_set = set(category_list)  # 提取该数据集各种类的值

        sum_num = 0
        for value in category_set:
            sum_num += (list(category_list).count(value) / float(len(data_set))) ** 2
        gini = 1 - sum_num
        return gini

    @staticmethod
    def tree_gini(tree, data_set, feature_name_list, gini):
        """
        :param tree: 当前子树
        :param data_set: 当前子树属的集合
        :param feature_name_list:特征名字列表
        :param gini:当前的基尼指数值
        :return: 当前子树的基尼指数
        """
        if type(data_set) == str:  # 单结点树（即以一个叶结点作为树）
            return CartTree.calculate_gini(data_set) * len(data_set)
        category_list = data_set.iloc[:, -1]  # 提取类别
        if len(set(category_list)) == 1:
            return 0  # 当前训练集的类别都一致
        if len(feature_name_list) == 1:  # 只剩下一个特征
            return CartTree.calculate_gini(data_set)
        node_feature = list(tree.keys())[0]  # 提取根结点
        loc_node_feat = feature_name_list.index(node_feature)  # 提取根结点所在列的位置
        feat_name_new = feature_name_list[:]  # 浅复制
        del feat_name_new[loc_node_feat]  # 删除该结点
        for value in list(tree[node_feature].keys()):
            if value != 'PreDivision' and value != '否':
                if value in list(data_set[loc_node_feat]):  # 数据集中能找到这个取值，说明可以划分
                    data_left, data_right = CartTree.split_data_for_cart(data_set, loc_node_feat, value)
                    # 按取值划分左右数据集（右支是'否'）
                    if type(tree[node_feature][value]) == dict:  # 不是叶结点，还需要递归
                        gini += CartTree.tree_gini(tree[node_feature][value], data_left, feat_name_new, gini)
                    else:  # 是叶结点，则计算其基尼指数
                        gini += len(data_left) * CartTree.calculate_gini(data_left)  # 这里用数据集的长度作为比重
                    # 下面对二叉树的另一部分求gini
                    if type(tree[node_feature]['否']) != dict:  # 是叶结点，则计算其基尼指数
                        gini += len(data_right) * CartTree.calculate_gini(data_right)  # 这里用数据集的长度作为比重
                    else:  # 不是叶结点，还需要递归
                        gini += CartTree.tree_gini(tree[node_feature]['否'], data_right, feat_name_new, gini)
                if value not in list(data_set[loc_node_feat]):  # 数据集中找不到这个取值
                    return 0
        return gini

    @staticmethod
    def tree_square_error(tree, data_set, feature_name_list, square_error):
        """
        :param tree: 当前子树
        :param data_set: 当前数据集
        :param feature_name_list: 当前特征列表
        :param square_error: 当前平方误差
        :return: 当前子树的平方误差
        """
        if type(tree) == int or type(tree) == float:  # 单结点树
            return ((np.array(data_set.iloc[:, -1]) - tree) ** 2).sum()  # 用矩阵计算数据集的平方误差
        node_feature = list(tree.keys())[0]  # 提取根结点
        pre_division_predict = tree[node_feature]['PreDivision']  # 提取欲划分的预测值
        if len(feature_name_list) == 1:  # 只剩下一个特征
            return ((np.array(data_set.iloc[:, -1]) - pre_division_predict) ** 2).sum()  # 用矩阵计算数据集的平方误差
        loc_node_feature = feature_name_list.index(node_feature)  # 根结点所在的位置
        feat_name_new = feature_name_list[:]
        del feat_name_new[loc_node_feature]
        for key in tree[node_feature].keys():  # 开始计算各结点的平方误差
            if key != 'PreDivision' and key != '否':
                if type(key) == tuple:  # key为元组，说明特征的取值为实数形式
                    value = key[1]  # 提取元组中的实数
                    data_left, data_right = CartTree.split_data_for_cart(data_set, loc_node_feature, value)
                elif key in list(data_set[loc_node_feature]):  # 数据集中有该取值（非实数型取值）
                    data_left, data_right = CartTree.split_data_for_cart(data_set, loc_node_feature, key)
                else:  # 数据集中没有该取值（非实数型取值）
                    continue
                # 下面开始计算每个叶结点平方误差
                if type(tree[node_feature][key]) == dict:  # 左支为子树
                    square_error += CartTree.tree_square_error(tree[node_feature][key], data_left, feat_name_new[:],
                                                               square_error)  # 对左支进行递归
                else:  # 左支为叶结点
                    prediction = tree[node_feature][key]  # 预测值
                    square_error += ((np.array(data_left.iloc[:, -1]) - prediction) ** 2).sum()
                if type(tree[node_feature]['否']) == dict:  # 右支为子树
                    square_error += CartTree.tree_square_error(tree[node_feature]['否'], data_right, feat_name_new[:],
                                                               square_error)  # 对右支进行递归
                else:  # 右支为叶结点
                    prediction = tree[node_feature]['否']  # 预测值
                    square_error += ((np.array(data_right.iloc[:, -1]) - prediction) ** 2).sum()
        return square_error

    @staticmethod
    def tree_cart_classification(data_set, features_name):
        """
        :param data_set: 当前数据集
        :param features_name: 含各特征名称的列表
        :return: CART分类树
        """
        category_list = data_set.iloc[:, -1]  # 提取类别
        if len(set(category_list)) == 1:
            return category_list[0]  # 当前训练集的类别都一致
        if len(features_name) == 1:  # 只剩下一个特征时无需划分
            return CartTree.majority_category(category_list)

        best_feature = features_name[0]
        best_value = data_set[0][0]
        min_gini = 9999
        for i in range(data_set.shape[1] - 1):  # 划分数据集和计算基尼指数
            feat_value_set = set(data_set[i])  # 提取该特征的不同取值
            for value in feat_value_set:
                data_left, data_right = CartTree.split_data_for_cart(data_set, i, value)
                left = len(data_left) / len(data_set) * CartTree.calculate_gini(data_left)
                right = len(data_right) / len(data_set) * CartTree.calculate_gini(data_right)
                gini = left + right
                if gini <= min_gini:
                    min_gini = gini
                    best_value = value
                    best_feature = features_name[i]

        pre_division_label = CartTree.majority_category(data_set.iloc[:, -1])  # 设置欲划分标签
        tree = {best_feature: {'PreDivision': pre_division_label}}  # 只写tree = {}就会报错
        loc_best_feature = features_name.index(best_feature)

        feat_name_new = features_name[:]
        del feat_name_new[loc_best_feature]
        sub_feat_name_left = feat_name_new[:]
        sub_feat_name_right = feat_name_new[:]

        data_left_new, data_right_new = CartTree.split_data_for_cart(data_set, loc_best_feature, best_value)
        if len(data_left_new) > 0:  # 不是空集
            tree[best_feature][best_value] = CartTree.tree_cart_classification(data_left_new, sub_feat_name_left)
        else:  # 空集
            tree[best_feature][best_value] = pre_division_label
        if len(data_right_new) > 0:  # 不是空集
            tree[best_feature]['否'] = CartTree.tree_cart_classification(data_right_new, sub_feat_name_right)
        else:  # 空集
            tree[best_feature]['否'] = pre_division_label
        return tree

    @staticmethod
    def tree_regression(data_set, features_name):
        if len(data_set) == 0:  # 空集
            return 0
        category_list = data_set.iloc[:, -1]  # 提取类别
        if len(set(category_list)) == 1:  # 类别全部一致
            return category_list[0]  # 当前训练集的类别都一致
        if len(features_name) == 1:  # 只剩下一个特征时无需划分
            return round(data_set.iloc[:, -1].mean(), 2)  # 返回平均值

        best_feature = features_name[0]
        best_val = float(data_set[:1][0])  # 特征的编号与其最佳划分点
        min_mean = 999
        for i in range(data_set.shape[1] - 1):  # 寻找最优切分点
            feat_value_set = set(data_set[i])  # 第i个特征的不同取值
            for value in feat_value_set:
                data_left, data_right = CartTree.split_data_for_cart(data_set, i, value)
                if data_left.shape[0] < 2 or data_right.shape[0] < 2:  # 样本数太少，不作划分
                    continue
                class_one = data_left.iloc[:, -1].mean()
                class_two = data_right.iloc[:, -1].mean()
                if (class_one + class_two) <= min_mean:
                    min_mean = class_one + class_two
                    best_val = value
                    best_feature = features_name[i]
        pre_division_label = round(data_set.iloc[:, -1].mean(), 2)  # 设置欲划分标签，方便剪枝时使用
        tree = {best_feature: {}}
        tree[best_feature]['PreDivision'] = pre_division_label
        loc_best_feat = features_name.index(best_feature)
        feat_name_new = features_name[:]
        del feat_name_new[loc_best_feat]

        sub_feat_name_left = feat_name_new[:]  # 左支的特征列表
        sub_feat_name_right = feat_name_new[:]  # 右支的特征列表
        data_new_left, data_new_right = CartTree.split_data_for_cart(data_set, loc_best_feat, best_val)

        if type(best_val) != int and type(best_val) != float:  # 特征的取值不是实数类型
            tree[best_feature][best_val] = CartTree.tree_regression(data_new_left, sub_feat_name_left)
            tree[best_feature]['否'] = CartTree.tree_regression(data_new_right, sub_feat_name_right)
        else:
            tree[best_feature]['<=', best_val] = CartTree.tree_regression(data_new_left, sub_feat_name_left)
            # 实数类型用元组储存
            tree[best_feature]['否'] = CartTree.tree_regression(data_new_right, sub_feat_name_right)
        return tree

    @staticmethod
    def convert_tree(tree_labeled):
        """
        :param tree_labeled: 带有'PreDivision'欲划分标签的树
        :return: 没有欲划分标签的树
        """
        if type(tree_labeled) != dict:  # 单结点树
            return tree_labeled
        node_feature = list(tree_labeled.keys())[0]  # 当前结点的名称
        tree_labeled_new = tree_labeled.copy()  # 浅复制
        tree_labeled_new[node_feature] = tree_labeled[node_feature].copy()  # 继续浅复制
        for value in list(tree_labeled_new[node_feature].keys()):
            if value == 'PreDivision':
                tree_labeled_new[node_feature].pop(value)
            elif type(tree_labeled_new[node_feature][value]) == dict:
                tree_labeled_new[node_feature][value] = CartTree.convert_tree(tree_labeled_new[node_feature][value])

        return tree_labeled_new

    @staticmethod
    def alpha_list_classification(alpha, tree, data_set, features_name):
        """
        :param alpha: 各结点的参数值列表(以字典形式存储)
        :param tree: 子树
        :param data_set:当前数据集
        :param features_name:当前特征列表
        :return: 最终返回整体树内各结点的参数
        """
        node_feature = list(tree.keys())[0]  # 提取根结点
        tree_new = tree.copy()  # 第一层浅复制
        tree_new[node_feature] = tree[node_feature].copy()  # 第二层浅复制
        del tree_new[node_feature]['PreDivision']  # 删掉欲划分标签
        loc_node_feature = features_name.index(node_feature)  # 当前结点所在列的位置
        feat_name_new = features_name[:]
        single_node_gini = CartTree.calculate_gini(data_set) * len(data_set)  # 单结点树的预测误差
        data_gini = CartTree.tree_gini(tree, data_set, feat_name_new, 0)  # 该特征作为根结点树的预测误差
        alpha[node_feature] = (single_node_gini - data_gini) / (CartTree.cal_leaves_number(tree, 0) - 1)  # 该结点的alpha

        for value in list(tree_new[node_feature].keys()):  # 寻找非叶结点并划分数据集
            if type(tree_new[node_feature][value]) == dict and value != '否':
                data_set_left, data_set_right = CartTree.split_data_for_cart(data_set, loc_node_feature, value)
                # 划分数据集
                del feat_name_new[loc_node_feature]
                alpha = CartTree.alpha_list_classification(alpha, tree_new[node_feature][value], data_set_left,
                                                           feat_name_new)  # 对左支递归
                if type(tree_new[node_feature]['否']) == dict:
                    alpha = CartTree.alpha_list_classification(alpha, tree_new[node_feature]['否'], data_set_right,
                                                               feat_name_new)  # 对右支递归
            elif type(tree_new[node_feature][value]) != dict and value != '否':
                if type(tree_new[node_feature]['否']) == dict:
                    data_set_left, data_set_right = CartTree.split_data_for_cart(data_set, loc_node_feature, value)
                    # 划分数据集
                    del feat_name_new[loc_node_feature]
                    alpha = CartTree.alpha_list_classification(alpha, tree_new[node_feature]['否'], data_set_right,
                                                               feat_name_new)  # 对右支递归
        return alpha

    @staticmethod
    def alpha_list_regression(alpha, tree, data_set, features_name):
        """
        :param alpha: 各结点的参数值列表(以字典形式存储)
        :param tree:  子树
        :param data_set:当前数据集
        :param features_name: 当前特征列表
        :return:  最终返回整体树内各结点的参数
        """
        node_feature = list(tree.keys())[0]  # 提取根结点
        tree_new = tree.copy()  # 第一层浅复制
        tree_new[node_feature] = tree[node_feature].copy()  # 第二层浅复制
        prediction = tree_new[node_feature].pop('PreDivision')  # 删掉欲划分标签并提取预测值
        loc_node_feature = features_name.index(node_feature)  # 当前结点所在列的位置
        feat_name_new = features_name[:]
        single_node_error = ((np.array(data_set.iloc[:, -1]) - prediction) ** 2).sum()  # 单结点树的平方误差
        node_feature_error = CartTree.tree_square_error(tree, data_set, features_name[:], 0)  # 当前结点作为根结点树的平方误差
        alpha[node_feature] = (single_node_error - node_feature_error) / (CartTree.cal_leaves_number(tree, 0) - 1)
        for key in list(tree[node_feature].keys()):  # 寻找下一个结点并划分数据集
            if key != '否':
                if type(tree[node_feature][key]) == dict or type(tree[node_feature]['否']) == dict:  # 左支或右支为子树
                    if type(key) == tuple:  # key为元组形式，说明取值为实数型
                        value = key[1]  # 用正则表达式提取其中的数字（字符串）并将其转化为浮点数
                        data_left, data_right = CartTree.split_data_for_cart(data_set, loc_node_feature, value)
                    elif key in list(data_set[loc_node_feature]):  # 数据集中有该取值（非实数型取值）
                        data_left, data_right = CartTree.split_data_for_cart(data_set, loc_node_feature, key)
                    else:
                        continue
                    del feat_name_new[loc_node_feature]
                    if type(tree[node_feature][key]) == dict:  # 左支是子树，对左支递归
                        alpha = CartTree.alpha_list_regression(alpha, tree[node_feature][key], data_left,
                                                               feat_name_new[:])
                    if type(tree[node_feature]['否']) == dict:  # 右支是子树，对右支递归
                        alpha = CartTree.alpha_list_regression(alpha, tree[node_feature]['否'], data_right,
                                                               feat_name_new[:])
        return alpha

    @staticmethod
    def puring(tree, best_feature):
        """
        :param tree: 子树
        :param best_feature: 需要进行剪枝的结点
        :return: 剪枝后的树
        """
        node_feature = list(tree.keys())[0]  # 该子树的根结点
        tree_new = tree.copy()  # 第一层浅复制
        tree_new[node_feature] = tree[node_feature].copy()  # 第二层浅复制
        if node_feature == best_feature:
            tree_new = tree_new[node_feature]['PreDivision']  # 找到对应结点，进行剪枝
            return tree_new
        else:
            for value in tree_new[node_feature].keys():
                if value == 'PreDivision':
                    continue
                elif type(tree_new[node_feature][value]) == dict:
                    tree_new[node_feature][value] = CartTree.puring(tree_new[node_feature][value], best_feature)
        return tree_new

    @staticmethod
    def get_sub_tree_sequence(tree, data_set, features_name, sub_tree_sequence, tree_type='class'):
        """
        :param sub_tree_sequence: 子树序列(用列表内嵌套字典的形式存储，每个字典代表每一颗子树)
        :param data_set:当前数据集
        :param features_name:当前特征列表
        :param tree: 当前子树
        :param tree_type:树的类型（默认为分类树）
        :return: 最终返回完整的子树序列
        """
        if len(sub_tree_sequence) == 0:
            sub_tree_sequence.append(tree)  # 把原始的树加进去
        alpha_list = {}  # 各结点的参数值列表
        if tree_type == 'class':  # 树的类型为分类树
            alpha_list = CartTree.alpha_list_classification(alpha_list, tree, data_set, features_name[:])
            # 得到该子树各结点对应的参数，以字典形式存储
        if tree_type == 'regression':  # 树的类型为回归树
            alpha_list = CartTree.alpha_list_regression(alpha_list, tree, data_set, features_name[:])
            # 得到该子树各结点对应的参数，以字典形式存储
        min_alpha = 99999
        best_feature = alpha_list.keys()
        for feature, alpha in alpha_list.items():  # 找到最小的alpha
            if alpha < min_alpha:
                best_feature = feature
                min_alpha = alpha
        tree_new = CartTree.puring(tree, best_feature)  # 对最佳结点进行剪枝
        sub_tree_sequence.append(tree_new)

        if type(tree_new) == dict and CartTree.cal_leaves_number(tree_new, 0) != 2:  # 继续剪枝，直到变成结构最简单的二叉树;
            sub_tree_sequence = CartTree.get_sub_tree_sequence(tree_new, data_set, features_name, sub_tree_sequence,
                                                               tree_type)
        return sub_tree_sequence

    @staticmethod
    def find_best_tree(sub_tree_sequence, data_set, features_name, tree_type='class'):
        """
        :param sub_tree_sequence: 子树序列
        :param data_set: 验证集
        :param features_name:特征列表
        :param tree_type:树的类型（默认为分类树）
        :return: 最优子树
        """
        lost = []  # 用于储存各颗子树的基尼指数
        for tree in sub_tree_sequence:
            if type(tree) == dict:
                if tree_type == 'class':  # 分类树算基尼指数
                    lost.append(CartTree.tree_gini(tree, data_set, features_name, 0))
                if tree_type == 'regression':  # 回归树算平方误差
                    lost.append(CartTree.tree_square_error(tree, data_set, features_name[:], 0))
            else:  # 单结点树
                if tree_type == 'class':  # 分类树算基尼指数
                    lost.append(CartTree.calculate_gini(data_set) * len(data_set))  # 由于前面的计算都乘了比重，这里也要乘
                if tree_type == 'regression':  # 回归树算平方误差(回归树的单结点树是一个实数)
                    lost.append(((np.array(data_set.iloc[:, -1]) - tree) ** 2).sum())
        min_lost = 999
        best_loc = 0
        loc = 0
        for num in lost:
            if min_lost >= num:  # 出现值相同的情况时选择后面的（树更简单）
                min_lost = num
                best_loc = loc
            loc += 1
        best_tree = sub_tree_sequence[best_loc]  # 最优子树
        return CartTree.convert_tree(best_tree)

    @staticmethod
    def cart_decision_tree(train_data, test_data, features_name, tree_type='class'):
        """
        :param train_data: 训练数据集
        :param test_data:测试数据集
        :param features_name: 特征列表
        :param tree_type:树的类型，默认为分类树
        :return: 经过剪枝得到的cart决策树
        """
        sub_tree = []
        feat_name_new = features_name[:]
        tree_pre_division = {}
        if tree_type == 'class':
            tree_pre_division = CartTree.tree_cart_classification(train_data, features_name)
        if tree_type == 'regression':
            tree_pre_division = CartTree.tree_regression(train_data, feat_name_new)
        sub_tree = CartTree.get_sub_tree_sequence(tree_pre_division, train_data, feat_name_new, sub_tree, tree_type)
        tree = CartTree.find_best_tree(sub_tree, test_data, feat_name_new)
        return tree

    @staticmethod
    def get_predict_single(tree, data_set_single, features_name, tree_type='class'):
        """
        :param tree: 决策树
        :param data_set_single: 单个样本
        :param features_name: 特征列表
        :param tree_type: 决策树类型，默认为分类树
        :return: 单个样本的预测值
        """
        node_feature = list(tree.keys())[0]
        loc_node_feature = features_name.index(node_feature)
        value = list(tree[node_feature].keys())[0]
        predict = tree[node_feature][value]
        judge = 1  # 用于判断取值在左支还是右支
        if tree_type == 'class':
            if value == list(data_set_single[loc_node_feature])[0]:  # 说明在左支
                judge = 1
            else:  # 说明在右支
                judge = 0
        elif tree_type == 'regression':
            num = value[1]  # 提取元组中的实数
            if list(data_set_single[loc_node_feature])[0] <= num:  # 说明在左支
                judge = 1
            else:  # 说明在右支
                judge = 0
        if judge == 1:
            if type(tree[node_feature][value]) == dict:  # 子树
                predict = CartTree.get_predict_single(tree[node_feature][value], data_set_single, features_name,
                                                      tree_type)
            else:  # 叶结点
                predict = tree[node_feature][value]
        elif judge == 0:
            if type(tree[node_feature]['否']) == dict:  # 子树
                predict = CartTree.get_predict_single(tree[node_feature]['否'], data_set_single, features_name,
                                                      tree_type)
            else:  # 叶结点
                predict = tree[node_feature]['否']
        return predict

    @staticmethod
    def get_predict(tree, data_set, features_name, tree_type='class'):
        """
        :param tree: 决策树
        :param data_set: 数据集
        :param features_name: 特征列表
        :param tree_type: 决策树类型，默认为分类树
        :return: 整个数据集的预测值
        """
        pre = []
        for i in range(len(data_set)):
            pre.append(CartTree.get_predict_single(tree, data_set[i:i + 1], features_name, tree_type))
        return pre
