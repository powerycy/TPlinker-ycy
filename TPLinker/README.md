# TPLINKER
TPLINKER注释版本，并适配了中文数据集<br>
中文数据来源于百度关系抽取大赛<br>
在preprocess路径下build_data_config.yaml中先配置数据源，注意ori_data_format因为用的自己的数据集所以为tplinker<br>
add_char_span设置为True方便添加char_span<br>
在ori_data/baidu_relation/data下dataprocess.py处理百度数据，只处理了一部分数据只是为了方便跑通，看效果<br>
在preprocess路径下运行BuildData.py生成数据，结果放置在data4bert/baidu_relation下<br>
在tplinker/train_config.yaml配置相应的文件<br>
接下来只需运行tplinker下的train.py即可运行。<br>
详解说明https://zhuanlan.zhihu.com/p/342300800<br>
最近因为工作原因，在搞模型压缩跟联邦学习，等到空闲了把细节补上。感谢~<br>

