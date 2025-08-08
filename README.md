# best_variant
针对MMA-DFER在AVEC2014上表现最好的迁移方法的变体与消融实验等
## 单模态与多模态对比
### video
The best Validation RMSE: 8.1672, MAE: 6.1775<br>
<img width="1590" height="789" alt="log" src="https://github.com/user-attachments/assets/bdfd90bb-acc2-4630-9545-23f7633a24bc" />
视觉模态在抑郁评估中特别有效，优于Baseline中所有模型，<br>
### audio
-------待完成-------<br>
## 消融实验
### no prompt
-------待完成-------<br>
### no fusion
The Best Validation RMSE: 8.7411, MAE: 6.5978<br>
<img width="1590" height="789" alt="log" src="https://github.com/user-attachments/assets/77978a76-e9e3-4d7a-9e3d-8cc7eedb9124" />
多模态融合后预测误差反而增大，按理说信息具有互补性，性能应该更好，结合前面复现的结果，猜测是模态融合部分选型不适合或者音频模块部分有问题，可进一步调整。<br>
### no temporal
The Best Validation RMSE: 9.6371, MAE: 8.0579<br>
<img width="1590" height="789" alt="log" src="https://github.com/user-attachments/assets/35188d14-c642-4c9e-97c6-bcb05a5b2dc3" />
拟合效果不好，要更换参数继续训练<br>
时序模块对于抑郁分数回归模型的性能至关重要，去除该模块之后性能降低<br>
## 使用
分别将nofusion、notemporal、video等文件夹的脚本替换根目录下的文件，而后执行训练与评估
