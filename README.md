#### 成功实现yolov5的目标检测以及realsense的深度信息返回
#### 成功实现ros发布节点
#### 成功将deep_sort封装近来

调用的时候初始化detect,dee_psort,ros.pubisher，将update以及发布方的publish函数以回调函数的方式传递

当有检测信息的时候调用publish发布出去

```
    det.action(pub.publish, deep_sort.update)
```


如果只是测试，可以在当前目录下运行`test01.py`
```
    python3 test01.py
```

如果在ros功能包下，请将`test01_ros.py`移动到ros功能包中
```
    cp test01_ros.py <your file directory>
```

修改`test01_ros.py`第5行`/home/adminpc/yolov5-deepsort-realsense-ros`为你的文件地址

然后运行

```
    python3 test01_ros.py
```


