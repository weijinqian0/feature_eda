# feature_eda
## 特征工程用到工具
整理所有特征工程用到的方法，方便复用

## 常用模型
添加常用模型

## 深度学习模型
默默地添加一下deepCTR的代码吧
谁让我自己不会写呢

## 从头开始一个模型
### 项目背景分析
1. 多个维度寻找数据；
2. 找出数据之间的联系与区别
### 数据分析
#### 数据质量分析
缺失值、异常值、重复值、歧义值、正负样本比例（样本不均衡）

#### 数据统计量分析
1. 单个变量的统计分析。直方图、箱型图、小提琴图
2. 两个变量分析。散点图、相关性分析图、热力图
3. 多个变量分析。彩色散点图

#### 数据分布分析
1. 频数维度:
不同点击次数下的用户分布（代码怎么写）
2. 时间维度统计分布：每天点击趋势
3. 空间维度统计分布：变量与地理位置有关

以上三种分析常常结合分组or聚类方法，对细分的业务场景进行考察，为数据建模做铺垫

#### 数据分析小结
1. 对比。训练集不同样本的特征分布。训练集和测试集的特征分布对比
2. 分组。常常按照类别标签、某个离散变量的不同取值groupby的sum、nunique
3. 频数。计算某些变量的累积分布。诸如“事件发生次数”这样的的统计量需要自己计算；有时还要关注“同id下某个事件多次发生”的统计。
4. 抓大放小。特征重要度高的变量，要重点分析。
5. 可视化。多画图

### 特征工程
#### 数据预处理
1. 数据清洗（缺失值、离群值、删除）
2. 数据集成（多表数据整合：一对一、一对多、多对一，多对多）
3. 数据重采样（滑窗法：时间序列。选取不同的时间窗间隔，可以得到多份训练数据集
该方法可以增加训练样本，也方便做交叉验证实验。非平衡重采样：调整正负样本量、欠采样、过采样、组合采样）
4. 数据变换（连续变量离散化、等频、等宽、聚类、离散变量编码、One-hot Encoding、Label Encoding、长尾分布、Ln、Log）
5. 数据规范化（Min-Max、Z-score、MaxAbs）
对应脚本（feature_transform base_utils）

#### 特征构造
##### 思想
1. 理解字段（应该就是各个表整合的过程）
2. 抽取实体（用户id、商品id、商家id）
3. 分析实体关系（用户id-商品id，用户id-商家id，商家id-商品id）
4. 设计特征群（用户维度：用商品、商家、点击来表征。商品维度：用户点击数分布、购买分布。商家维度：产品数、用户点击数、用户购买数、商品上线时间）
5. 按特征群分别构造特征（分别按照以上来构造特征）

#### 特征选择
1. 过滤式选择；
2. 包装式选择；
3. 嵌入式选择；

文件位置：feature_filter feature_embedded feature_wrapper

参数学习
parameter_search

### 模型优化
![](./model_optimization.jpg)
#### 评估方法
留出法，自助法和交叉验证法

#### 性能度量
AUC、Logloss、K-S、F1、Kappa系数

#### 参数调整
网格搜索（Grid Search）
贝叶斯优化（Bayesian Optimization）
启发式算法（Heuristic Algorithms，如GA、模拟退火，PSO）
工具包：Hyperopt

### 模型融合
1. 简单加权平均：0.5*result_1+0.5*result_2
2. Bagging：对训练集随机采样，训练不同的base model，然后投票；可以减少方差，提升模型的稳定性（随机森林就是这个原理）
3. Boosting：弱分类器提升为强分类器，并做模型的加权融合；可以减少学习误差，但容易过拟合
4. Blending：拆分训练集，使用不重叠或者部分重叠的数据训练不同的base model，然后分别预测test数据，并加权融合（这是个好办法）
5. Stacking：网上讲的很多，但极易造成过拟合，尤其是数据量小时过拟合严重，不建议使用
