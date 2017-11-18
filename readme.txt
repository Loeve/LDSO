LDSO-2017.11.9：
debug完成最初版本

LDSO-2017.11.13-all-pic：
一次性处理test_images中所有图片，结果保存在result_map中

LDSO-2017.11.13-one-pic：
打开任意路径图片处理，结果保存在result_map中

文件说明：
cython优化：优化加速用过的pyx和相应的setup文件，可进一步改动进行优化
ground_truth:自然图像的显著图参考标准
result_map：保存运行后结果图
test_images:所有要测试的图像
main.py：主函数
extract_feature_maps.py:特征提取函数（部分）
infer_salient_map.py：显著图计算函数（部分）
cython加速文件：
EFM.cp35-win_amd64.pyd
Infer_map.cp35-win_amd64.pyd
bp.cp35-win_amd64.pyd
