# 📊 直播带货成交额预测模型

## 📌 项目简介
使用机器学习线性回归模型，基于直播间转发量等特征预测商品成交额，为直播运营提供数据支持。

## 🛠️ 环境要求
```bash
Python 3.6+
pip install pandas matplotlib seaborn scikit-learn
```

## 核心算法

# 数据预处理
X = df[['转发量']]  # 特征矩阵
y = df['成交额']    # 目标变量

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=0
)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测评估
y_pred = model.predict(X_test)


## 输出结果

✅ 数据加载成功！样本数：1000
📈 模型性能评估：
训练集R²得分：0.85
测试集R²得分：0.82

## 参数配置

参数	             类型	   默认值	    说明
test_size	     float	   0.2	        测试集比例
random_state	 int	   0	        随机种子
font_family	     str	   'Heiti TC'	中文字体


## 常见问题

中文显示乱码？

确保系统安装黑体字体

修改代码中的plt.rcParams['font.sans-serif']

如何增加新特征？

df['互动率'] = df['转发量'] / df['观看人数']

