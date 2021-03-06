# -binning-algorithms-
特征工程之特征分箱：决策树分箱、卡方分箱、bestks以及评价标准
前言：在做数据挖掘项目的时候，特征工程通常是其中非常重要的一个环节，但是难度也比较高，并且不同项目特征工程会有所差异，因此在做相关项目时可以多总结一些对结果提升比较明显的操作，在做金融风控项目时，发现在对部分连续变量和离散变量进行分箱之后和woe编码后效果有一定提升。
在特征工程我们通常可以对连续值属性进行分箱操作（也就是常说的离散化），并且对于取值较多的离散变量也可进行分箱操作，分箱之后主要有以下好处：
1.分箱后的特征对异常数据有更好的鲁棒性。例如年龄中含有200，300这类的数据，分箱之后就可能划到>80这一箱中，但是如果直接传入模型训练的话会对模型造成很大干扰。
2.可以将变量转换到相似的尺度上。例如收入有1000，10000，百万，千万等，可以离散化为0（低收入），1（中等收入），2（高收入）等。
3.离散后的变量方便我们做特征交叉。例如将离散后的特征与连续变量进行groupby，构造均值、方差等。
4.缺失值的处理，可以将缺失值合并为一个箱。
并且分箱我们通常会遵循以下原则：
1.组内差异小
2.组间差异大
3.每组占比不小于5%
4.必须有好坏两种分类（对于二分类而言）
对于某个属性分箱过后的结果是好还是坏，我们可以使用WOE和IV进行评估。

1.WOE和IV
（1）WOE（Weight Of Evidence），即证据权重，
（2）IV（Information Value），是用来衡量某一变量的信息量。其计算方式如下：
woe和iv的python实现。

知道了特征分箱后的评估方法之后就可以考虑如何进行分箱。而分箱的方法主要分为两大类：无监督分箱（等频分箱、等距分箱），有监督分箱（best-ks分箱、决策树分箱、卡方分箱）。

2.无监督分箱
2.1等频分箱
等频分箱的意思就是分箱之后，每个箱内的数据量相等。可以用pandas提供的qcut方法进行处理。

2.2等距分箱
等距分箱就是每个箱的区间的大小是相等的，每个箱内的数据量不一定相等。

3.有监督分箱
3.1决策树分箱
决策树分箱的原理就是用想要离散化的变量单变量用树模型拟合目标变量，例如直接使用sklearn提供的决策树（是用cart决策树实现的），然后将内部节点的阈值作为分箱的切点。
补充，cart决策树和ID3、C4.5决策树不同，cart决策树对于离散变量的处理其实和连续变量一样，都是将特征的所有取值从小到大排序，然后取两两之间的均值，然后遍历所有这些均值，然后取gini系数最大的点作为阈值进行划分数据集。并且该特征后续还可参与划分。

3.2best-ks分箱
KS（Kolmogorov Smirnov）用于模型风险区分能力进行评估，指标衡量的是好坏样本累计部分之间的差距 。KS值越大，表示该变量越能将正，负客户的区分程度越大。通常来说，KS>0.2即表示特征有较好的准确率。

KS的计算方式：计算分箱后每组的好账户占比和坏帐户占比的差值的绝对值：∣ g o o d 占 比 − b a d 占 比 ∣ |good占比-bad占比|∣good占比−bad占比∣，然后取所有组中最大的KS值作为变量的KS值。

best-ks分箱的原理：

将变量的所有取值从小到大排序
计算每一点的KS值
选取最大的KS值对应的特征值x m x_mx 
m
​	
 ，将特征x分为x i ≤ x m {x_i\leq x_m}x 
i
​	
 ≤x 
m
​	
 和x i > x m {x_i>x_m}x 
i
​	
 >x 
m
​	
 两部分。然后对于每部分重复2-3步。

3.3卡方分箱
卡方检验的原理：卡方分箱是依赖于卡方检验的分箱方法，在统计指标上选择卡方统计量（chi-Square）进行判别，分箱的基本思想是判断相邻的两个区间是否有分布差异，基于卡方统计量的结果进行自下而上的合并，直到满足分箱的限制条件为止。
基本步骤：
预先设定一个卡方的阈值
对需要进行离散的属性进行排序，每个取值属于一个区间
合并区间：(1)计算每一对相邻区间的卡方值：
 ；(2)然后将卡方值最小的一对区间合并。
上述步骤的终止条件：
分箱个数：每次将样本中具有最小卡方值的区间与相邻的最小卡方区间进行合并，直到分箱个数达到限制条件为止。
卡方阈值：根据自由度和显著性水平得到对应的卡方阈值，如果分箱的各区间最小卡方值小于卡方阈值，则继续合并，直到最小卡方值超过设定阈值为止。
