import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def plot_matrix(conf_matrix, name):
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    class_accuracy = conf_matrix / conf_matrix.sum(axis=1)[:, None]
    plt.imshow(class_accuracy, cmap=plt.get_cmap('Blues'))
    plt.grid(False)
    plt.colorbar()
    axis_label = np.array(['0', '1', '2'])
    num_local = np.array(range(len(axis_label)))
    plt.xticks(num_local, axis_label, fontsize=12)
    plt.yticks(num_local, axis_label, fontsize=12)
    thresh = 0.5
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.text(j, i, '{:.2f}'.format(class_accuracy[i][j] * 100) + '%', ha="center", va="center",
                     color="black" if conf_matrix[i][j] > thresh else "while")
    plt.savefig(name + 'Confusion_Matrix' + '.png')
    plt.close()


def evaluate(y_true, y_pred, name, draw=False):
    # print(y_true[100:200])
    # print(y_pred[100:200])
    # 导入混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 计算精确率
    precision = precision_score(y_true, y_pred, average='micro')
    # 计算召回率
    recall = recall_score(y_true, y_pred, average='micro')
    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='micro')
    # 打印性能指标
    print("混淆矩阵:")
    print(conf_matrix)
    print("准确率:", accuracy)
    print("精确率:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)

    if draw:
        plot_matrix(conf_matrix, name)
