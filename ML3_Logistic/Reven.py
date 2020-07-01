# _*_ coding:utf-8 _*_
# python3.8
# writer:lierdere
# 2020-06

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# 训练函数
def train(x_train, y_train, epoch, learning_rate):
    num = x_train.shape[0] #读取训练集行数=3000
    dim = x_train.shape[1] #读取训练集列数=57
    bias = 0 #初始化偏置值
    weights = np.ones(dim) #初始化权重
    reg_rate = 0.000 #正则项系数
    bg2_sum = 0 #用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)#用于存放权重的梯度平方和
    
    Losss = np.zeros(int(epoch/2)) #存放每一轮训练后的损失值
    Accs  = np.zeros(int(epoch/2)) #存放每一轮训练后的准确率
    bg_s  = np.zeros(int(epoch/2)) #存放每一轮训练后更新的偏置常数
    wg_s  = np.zeros((int(epoch/2),dim)) #存放每一轮训练后更新的权重
    m = 0 #计数器

    for i in range (epoch):
        b_g = 0 #偏置值梯度
        w_g = np.zeros(dim) #权重值梯度
        #在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias #线性回归
            sig = 1 / (1 + np.exp(-y_pre)) #压缩回归值，得到概率值
            b_g += (-1) * (y_train[j] - sig) #损失函数对偏置值求导
            for k in range(dim):
                #对权重w求梯度，2reg_ rateweights[k]为正则项，防止过拟合
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j,k] + 2*reg_rate * weights[k]
        b_g /= num
        w_g /= num
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        #更新权重和偏置常数
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum **0.5 *w_g
 
        if i % 2 == 0:
            acc = 0
            loss = 0
            result = np.zeros(num)
            for j in range(num):
                y_pre = weights.dot(x_train[j, :]) + bias
                sig = 1 / (1 + np.exp(-y_pre))
                if sig >= 0.5:
                    result[j] = 1
                else:
                    result[j] = 0

                if result[j] == y_train[j]:
                    acc += 1.0
                #为了避免sig无限接近于1,导致计算出错
                #最终优化
                loss += y_train[j] * np.log(sig**(-1)) + (1 - y_train[j]) * (np.log(1 + np.exp(-y_pre))+ y_pre)
            #保存每一轮训练后的准确率和损失率
            Losss[m] = loss / num
            Accs[m]  = acc / num
            bg_s[m]  = bias
            wg_s[m]  = weights 
            m += 1
    return  weights, bias,Losss,Accs,wg_s,bg_s
# 验证函数
def validate(x_val,y_val,weights,bias):
    num = 1000 #样本数
    loss = 0 #初始化损失值
    acc = 0  #初始化准确度
    result = np.zeros(num)
    for j in range(num):
        y_pre = weights.dot(x_val[j,:]) + bias #对第j个样本回归
        sig = 1 / (1 + np.exp(-y_pre)) #压缩回归值 
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0
        #若预测值和实际值相同，计数器加1
        if result[j] == y_val[j]:
            acc += 1.0
        loss += y_val[j] * np.log(sig**(-1)) + (1 - y_val[j]) * (np.log(1 + np.exp(-y_pre))+ y_pre)
    return acc / num, loss / num #返回准确度和loss的平均值
#绘图函数
def draw(x_data,y1_data,y2_data):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('times')
    ax1.set_ylabel('Acc', color=color)
    ax1.plot(x_data, y1_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()   

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)   
    ax2.plot(x_data, y2_data, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()   
    plt.show()
   
def draw1(x_data,y1_data,y2_data):
    plt.plot(x_data, y1_data, linewidth=0.6, label = 'Acc')
    plt.plot(x_data, y2_data, linewidth=0.6, label = 'Loss')
    plt.legend()
    plt.xlabel('Learning_rate')
    plt.ylabel('Vlue')
    plt.title('Learning_rate')
    plt.show()

def  main():
    # 读取文件信息
    all = pd.read_csv('income.csv')
    all =all.fillna(0) #空置处填0
    array = np.array(all)
    x = array[:, 1:-1] #取第1列到倒数第2列
    x[:, -1] /= np.mean(x[:, -1])  #倒数第2列规整化
    x[:, -2] /= np.mean(x[:, -2])  #倒数第3列规整化
    y = array[:,-1] #读取标签值
    #划分训练集与验证集。
    x_train, x_val = x[0:3000, :], x[-1000: , :] #各项属性
    y_train, y_val = y[0:3000], y[-1000: ]#标签值
    print('【1】观察训练轮数影响    【2】观察学习率影响')
    choice = int(input('请输入序号选择对应操作：'))
    if choice == 1:
        epoch = int (input('请输入最大训练轮数：'))
        learning_rate = float (input('请输入固定学习率：'))
        # 开始训练
        x_data = np.arange(0, epoch, 2)
        Losss = np.zeros(int (epoch / 2))
        Accs = np.zeros(int (epoch / 2))
        

        w,b,l,a,ws,bs = train(x_train, y_train, epoch,learning_rate)
        #验证集上看效果
        #for i in x_data:
        i = 0
        while i < int(epoch/2):
           Accs[i],Losss[i] = validate(x_val, y_val, ws[i], bs[i])
           print('训练',i,'轮后' ,'准确率为', format(Accs[i],'.3f'),'损失值为',Losss[i])
           i = i+1
        draw(x_data,a,l)
        draw(x_data,Accs,Losss)
    if choice == 2:
         epoch = int (input('请输入训练轮数：'))
         L_rate = float (input('请输入最小学习率(非负数）：'))
         M_rate = float (input('请输入最大学习率(建议不超过2.5）：'))
         Losss = np.zeros(int((M_rate-L_rate)/0.1)) #开辟相应数量数组，存储损失值
         Accs = np.zeros(int((M_rate-L_rate)/0.1))
         m = 0 #初始化计数器
         x = np.arange(L_rate,M_rate,0.1)
         for i in x:
             w,b,l,a,ws,bs = train(x_train, y_train, epoch,i)
             Accs[m],Losss[m] = validate(x_val, y_val, w, b)
             print('当学习率为',round(i,1),'准确度', format(Accs[m],'.3f'),'损失值为：',Losss[m])
             m += 1
             
         draw(x,Accs,Losss)
            
if __name__ == '__main__':
    main()



      

