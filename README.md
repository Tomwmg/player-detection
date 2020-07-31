# player-detection
a player dectection method based on multiple instance learning

这是一个基于多示例学习的运动员定位方法，其原理示意如下：
![image](https://github.com/Tomwmg/player-detection/blob/master/framework.jpg)

该方案基于通用检测器对人体的检测结果，将包含运动员的图片的检测结果作为正包，不包含运动员的图片的检测结果作为负包，然后对这些检测框进行特征提取，并将这些特征通过线性变换映射到同一个特征空间。初始化一个特征向量作为运动员特征，并计算所有检测框特征与运动员特征的相似度，取正包中检测框特征与运动员特征相似度的最大值作为正包相似度，同样取负包中相似度最大值作为负包相似度，依据正包相似度要大于负包相似度构建损失函数进行训练。训练完成后，计算检测结果对应的检测框特征与运动员特征的相似度，设定一个相似度阈值，将大于该阈值的检测结果作为运动员的检测结果。

**文件说明**

neg.npy文件是800×2048维的负包特征，pos.npy文件是800×2048维的正包特征（训练样例数据仅为演示demo，可以根据需要构建不同的正负包）

demo.pth是基于该正负特征训练的运动员特征区分模型。

**使用说明**

Python版本3.6以上，Pytorch版本0.4.0以上

运行方式：Python xx.py

**以下是基于Faster-RCNN通用检测模型的检测效果**

<img src="https://github.com/Tomwmg/player-detection/blob/master/base.jpg" width="500" height="300" alt="通用检测"/><br/>

<img src="https://github.com/Tomwmg/player-detection/blob/master/mil.jpg" width="500" height="300" alt="运动员检测"/><br/>


**参考文献**

[1]	Maron O, Lozano-Pérez T. A framework for multiple-instance learning[C]. Advances in neural information processing systems. 1998: 570-576.

[2]	Zhou L, Louis N, Corso J J. Weakly-supervised video object grounding from text by loss weighting and object interaction[J]. arXiv preprint arXiv:1805.02834, 2018.

[3]	Shi J, Xu J, Gong B, et al. Not all frames are equal: Weakly-supervised video grounding with contextual similarity and visual clustering losses[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 10444-10452.

[4]	Huang D A, Buch S, Dery L, et al. Finding[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5948-5957.

